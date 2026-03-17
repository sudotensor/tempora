from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/j-pong/CMF
# Paper : https://openreview.net/forum?id=BllUWdpIOA
# Note  : This version largely follows the code structure found in eta.py. It operates in float32 (no half-precision).
#         BN layers track running statistics (resetting source stats.) to enable frozen inference, unlike the original
#         which discards them. Only the 'lp' posterior mode is supported. The TTA augmentation for the consistency loss
#         is hardcoded to 224×224 (standard ImageNet resolution).
class CMF(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1, momentum_probs=0.9,
                 temperature=1/3, alpha=0.99, gamma=0.99, q=0.005):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.momentum_probs = momentum_probs  # EMA momentum for running class probability estimate
        self.temperature = temperature        # Sharpening temperature for the combined diversity-certainty weight
        self.alpha = alpha                    # Kalman predict momentum: how strongly the hidden state reverts toward source
        self.gamma = gamma                    # Ensemble momentum: how strongly the adapted model is nudged toward hidden state

        self.q = q  # Process noise scale: controls how fast the Kalman gain responds to drift

        # Kalman filter state
        self.hidden_var = 0.0

        # Running class probability estimate for diversity weighting; shape inferred on first forward pass
        self.class_probs_ema = None

        # TTA augmentation for consistency loss: applied per-sample to normalised image tensors
        self.tta_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
        ])

        # Source model: frozen reference for the Kalman filter
        self.src_model = deepcopy(self.model)
        for p in self.src_model.parameters():
            p.detach_()

        # Hidden model: the Kalman filter's running state estimate
        self.hidden_model = deepcopy(self.model)
        for p in self.hidden_model.parameters():
            p.detach_()

        # All three models start from the same state; save one copy for reset
        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                probs = outputs.softmax(1)

                with torch.no_grad():
                    if self.class_probs_ema is None:
                        self.class_probs_ema = torch.full((probs.shape[1],), 1.0 / probs.shape[1], device=probs.device)

                    # Diversity weight: high for samples differing from the running class distribution
                    div = 1 - F.cosine_similarity(self.class_probs_ema.unsqueeze(0), probs, dim=1)
                    div = (div - div.min()) / (div.max() - div.min())
                    mask = div < div.mean()  # Exclude samples with below-average diversity

                    # Certainty weight: high for confident (low-entropy) predictions
                    cert = (probs * outputs.log_softmax(1)).sum(1)  # = -H(p)
                    cert = (cert - cert.min()) / (cert.max() - cert.min())

                    weights = torch.exp(div * cert / self.temperature)
                    weights[mask] = 0.0

                    self.class_probs_ema = (self.momentum_probs * self.class_probs_ema
                                            + (1 - self.momentum_probs) * probs.detach().mean(0))

                # SLR loss (per-sample): more resistant to degenerate collapse than entropy
                probs_slr = probs.clamp(max=0.99)
                loss_slr = -(probs_slr * torch.log(probs_slr / (1.0 - probs_slr) + 1e-5)).sum(1)
                if not (~mask).any():
                    self.optimizer.zero_grad()
                    return outputs, prediction_time
                loss_slr = (loss_slr * weights)[~mask]
                n = x.shape[0]  # Full batch size: used as a constant denominator to keep gradient scale consistent
                loss = loss_slr.sum() / n

                # Consistency loss: diverse-filtered samples should predict similarly under augmentation
                x_aug = torch.stack([self.tta_transform(img) for img in x[~mask]])
                outputs_aug = self.model(x_aug)
                p, p_aug = outputs[~mask].softmax(1), outputs_aug.softmax(1)
                loss_sce = (-0.5 * (p * outputs_aug.log_softmax(1)).sum(1)
                            - 0.5 * (p_aug * outputs[~mask].log_softmax(1)).sum(1))
                loss = loss + (loss_sce * weights[~mask]).sum() / n

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Prior correction: scale logits by batch-level class frequencies (for prediction only)
            with torch.no_grad():
                prior = probs.detach().mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / prior.max()
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs = outputs.detach() * smoothed_prior

            self._bayesian_filter()
            return outputs, prediction_time

        if self.frozen:
            with torch.no_grad():
                return stopwatch(device, lambda: self.model(x).softmax(1))

        # Adaptation time includes the cost of the initial forward pass
        (outputs, prediction_time), adaptation_time = stopwatch(device, _forward)

        if self.reforward:
            with torch.no_grad():
                self.freeze()
                outputs, reforward_time = stopwatch(device, lambda: self.model(x))
                self.unfreeze()
                prediction_time = adaptation_time + reforward_time  # Full latency: forward + adapt + reforward

        return outputs.softmax(1), prediction_time

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.hidden_model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

        self.hidden_var = 0.0
        if self.class_probs_ema is not None:
            self.class_probs_ema.fill_(1.0 / self.class_probs_ema.shape[0])

    def freeze(self):
        self.frozen = True
        self.model.eval()

    def unfreeze(self):
        self.frozen = False
        self.model.train()

    @staticmethod
    def collect_params(model):
        ps = []
        ns = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        ps.append(p)
                        ns.append(f"{nm}.{np}")
        return ps, ns

    @torch.no_grad()
    def _bayesian_filter(self):
        # 1. Predict: pull hidden state toward source (mean-reversion prior)
        self._moments(self.hidden_model, self.src_model, self.alpha, update_all=True)

        # 2. Compute Kalman gain from accumulated variance
        self.hidden_var = self.alpha ** 2 * self.hidden_var + self.q
        r = 1.0 - self.q
        beta = r / (self.hidden_var + r)
        beta = 1.0 if beta >= 0.9999 else max(0.89, beta)  # Saturate to 1.0 (no-op in _moments) rather than cap
        self.hidden_var = beta * self.hidden_var

        # 3. Update hidden state toward gradient-updated model
        self._moments(self.hidden_model, self.model, beta, update_all=True)

        # 4. Ensemble: nudge adapted model toward filtered hidden state (trainable params only)
        self._moments(self.model, self.hidden_model, self.gamma)

    @staticmethod
    @torch.no_grad()
    def _moments(target, source, alpha, update_all=False):
        """EMA blend: target = alpha * target + (1 - alpha) * source. No-op when alpha >= 1.0."""
        if alpha >= 1.0:
            return
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            if t_param.requires_grad or update_all:
                t_param.data.mul_(alpha).add_((1.0 - alpha) * s_param.data)

    def _configure_model(self, model, momentum):
        model.train()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.requires_grad_(True)
                m.momentum = momentum
                m.reset_running_stats()  # Discard source statistics
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "CMF requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "CMF needs some parameters with gradients."
        assert not all(param_grads), "CMF should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
                       for m in model.modules())
        assert has_norm, "CMF requires normalization layers."
