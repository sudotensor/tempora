from math import log

import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/mr-eggplant/SAR
# Paper : https://arxiv.org/abs/2302.12400
# Note  : This version largely follows the code structure found in in tent.py. The momentum parameter is only for BN
#         layers.
class SAR(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1, e_threshold=0.4 * log(1000), r_threshold=0.2):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.e_threshold = e_threshold
        self.r_threshold = r_threshold

        self.ema_entropy = None

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            with torch.enable_grad():
                # First forward: Keep low-entropy, confident samples (reliability filter)
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                mask_1 = torch.where(entropy < self.e_threshold)
                loss = entropy[mask_1].mean(0)
                loss.backward()

                self.optimizer.first_step(zero_grad=True)  # Perturb weights to w' = w + ε(w)

                # Second forward: Check which samples remain reliable at perturbed weights
                outputs_2 = self.model(x)
                entropy_2 = -(outputs_2.softmax(1) * outputs_2.log_softmax(1)).sum(1)
                mask_2 = torch.where(entropy_2[mask_1] < self.e_threshold)
                loss_second = entropy_2[mask_1][mask_2].mean(0)

                if not torch.isnan(loss_second):
                    if self.ema_entropy is None:
                        self.ema_entropy = loss_second.item()
                    else:
                        self.ema_entropy = 0.9 * self.ema_entropy + 0.1 * loss_second.item()

                loss_second.backward()
                self.optimizer.second_step(zero_grad=True)  # Restore w and update using gradient at w'

                # Check for model recovery
                if self.ema_entropy is not None and self.ema_entropy < self.r_threshold:
                    self.reset()

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
        self.ema_entropy = None

    def freeze(self):
        self.frozen = True
        self.model.eval()

    def unfreeze(self):
        self.frozen = False
        self.model.train()

    @staticmethod
    def collect_params(model):
        # Skip top layers for adaptation: layer4 for ResNets and blocks 9-11 for ViT-Base
        skip_in_name = ["layer4", "blocks.9", "blocks.10", "blocks.11", "norm."]  # Substring match
        skip_is_name = ["norm"]  # Exact match

        ps = []
        ns = []
        for nm, m in model.named_modules():
            if nm in skip_is_name or any(s in nm for s in skip_in_name):
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
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
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "SAR requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "SAR needs some parameters with gradients."
        assert not all(param_grads), "SAR should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules())
        assert has_norm, "SAR requires normalization layers."
