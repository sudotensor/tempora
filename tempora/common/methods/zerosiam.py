import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/Cascol-Chen/ZeroSiam
# Paper : https://arxiv.org/abs/2412.02779
# Note  : This version largely follows the code structure found in spa.py, as both methods use FeatureAdapter.
#         The "predictor" in the original code corresponds to our "projector" in FeatureAdapter. Both optimizers
#         (norm layers and projector) are fully reset between distributions, unlike the original which only resets
#         the projector.
class ZeroSIAM(Method):
    def __init__(self, model, backbone_optimizer, projector_optimizer, reforward=False, momentum=0.1):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.backbone_optimizer = backbone_optimizer
        self.projector_optimizer = projector_optimizer
        self.reforward = reforward

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.backbone_optimizer_state = get_cpu_snapshot(self.backbone_optimizer.state_dict())
        self.projector_optimizer_state = get_cpu_snapshot(self.projector_optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                outputs_noisy = self.model(x, use_projector=True)

                # Entropy minimisation on the projector-perturbed view
                loss_ent = -(outputs_noisy.softmax(1) * outputs_noisy.log_softmax(1)).sum(1).mean()

                # Symmetric KL between clean and noisy views: prevents the projector from collapsing
                log_p = outputs.detach().log_softmax(1)
                log_q = outputs_noisy.log_softmax(1)
                p = log_p.exp()
                q = log_q.exp()
                loss_skl = ((p * (log_p - log_q)) + (q * (log_q - log_p))).sum(1).mean()

                loss = loss_ent + loss_skl
                loss.backward()

                self.backbone_optimizer.step()
                self.backbone_optimizer.zero_grad()
                self.projector_optimizer.step()
                self.projector_optimizer.zero_grad()

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
        self.backbone_optimizer.load_state_dict(self.backbone_optimizer_state)
        self.projector_optimizer.load_state_dict(self.projector_optimizer_state)

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
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        ps.append(p)
                        ns.append(f"{nm}.{np}")
        return ps, ns

    def _configure_model(self, model, momentum):
        model.train()
        model.requires_grad_(False)

        if hasattr(model, "projector"):
            model.projector.requires_grad_(True)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.momentum = momentum
                m.reset_running_stats()  # Discard source statistics
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "ZeroSIAM requires train mode."
        assert hasattr(model, "projector"), "ZeroSIAM requires a model with a projector (wrap with FeatureAdapter)."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "ZeroSIAM needs some parameters with gradients."
        assert not all(param_grads), "ZeroSIAM should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "ZeroSIAM requires normalization layers."
