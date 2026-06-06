from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Rebuttal ablation: ETA with source BN running statistics preserved (no reset_running_stats()).
# Identical to ETA in every other respect. Purpose: isolate whether ETA's reset of source-domain
# BN statistics is responsible for its instability at tighter amortised budgets.
class ETANoReset(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1, e_threshold=0.4 * log(1000), d_threshold=0.05):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.e_threshold = e_threshold
        self.d_threshold = d_threshold

        self.mm_probs = None
        self.num_reliable_samples = 0
        self.num_required_samples = 0

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

                mask = entropy < self.e_threshold
                if not mask.any():
                    self.optimizer.zero_grad()
                    return outputs.softmax(1), prediction_time

                entropy_filtered = entropy[mask]
                outputs_filtered = outputs[mask]
                self.num_reliable_samples += entropy_filtered.size(0)

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

                weight = torch.exp(self.e_threshold - entropy_filtered.detach()).requires_grad_(False)
                loss = (entropy_filtered * weight).mean(0)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            return outputs, prediction_time

        if self.frozen:
            with torch.no_grad():
                return stopwatch(device, lambda: self.model(x).softmax(1))

        (outputs, prediction_time), adaptation_time = stopwatch(device, _forward)

        if self.reforward:
            with torch.no_grad():
                self.freeze()
                outputs, reforward_time = stopwatch(device, lambda: self.model(x))
                self.unfreeze()
                prediction_time = adaptation_time + reforward_time

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
                # Source running statistics are intentionally preserved (no reset_running_stats())
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "ETANoReset requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "ETANoReset needs some parameters with gradients."
        assert not all(param_grads), "ETANoReset should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "ETANoReset requires normalization layers."
