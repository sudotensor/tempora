from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/mr-eggplant/EATA
# Paper : https://arxiv.org/abs/2204.02610
# Note  : This version implements ETA (the variant of EATA without anti-forgetting regularisation) and largely follows
#         the code structure found in tent.py. The momentum parameter is only for BN layers.
class ETA(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1, e_threshold=0.4 * log(1000), d_threshold=0.05):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.e_threshold = e_threshold  # Entropy threshold to filter out unreliable samples
        self.d_threshold = d_threshold  # Diversity threshold (cosine similarity) to filter out redundant samples

        self.mm_probs = None           # Moving mean probabilities
        self.num_reliable_samples = 0  # Reliable sample count
        self.num_required_samples = 0  # Non-redunant and reliable sample count

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)  # Conditional entropy

                # 1. Reliability filter: Keep low-entropy, confident samples
                mask = entropy < self.e_threshold
                if not mask.any():
                    self.optimizer.zero_grad()
                    return outputs.softmax(1), prediction_time

                entropy_filtered = entropy[mask]
                outputs_filtered = outputs[mask]
                self.num_reliable_samples += entropy_filtered.size(0)

                # 2. Redundancy filter: Keep samples dissimilar from the moving mean
                mask = torch.ones_like(entropy_filtered, dtype=torch.bool)
                if self.mm_probs is None:
                    with torch.no_grad():
                        self.mm_probs = outputs_filtered.softmax(1).mean(0)
                else:
                    cos_sims = F.cosine_similarity(self.mm_probs.unsqueeze(0), outputs_filtered.softmax(1), dim=1)
                    mask = cos_sims.abs() < self.d_threshold

                    if not mask.any():
                        self.optimizer.zero_grad()
                        return outputs.softmax(1), prediction_time

                    entropy_filtered = entropy_filtered[mask]
                    outputs_filtered = outputs_filtered[mask]

                    with torch.no_grad():
                        self.mm_probs = 0.1 * outputs_filtered.softmax(1).mean(0) + 0.9 * self.mm_probs

                self.num_required_samples += entropy_filtered.size(0)

                # 3. Compute weighted entropy loss
                weight = torch.exp(self.e_threshold - entropy_filtered.detach()).requires_grad_(False)
                loss = (entropy_filtered * weight).mean(0)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

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
        self.optimizer.load_state_dict(self.optimizer_state)

        self.mm_probs = None
        self.num_reliable_samples = 0
        self.num_required_samples = 0

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

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.momentum = momentum
                m.reset_running_stats()  # Discard source statistics
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "ETA requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "ETA needs some parameters with gradients."
        assert not all(param_grads), "ETA should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "TENT requires normalization layers."
