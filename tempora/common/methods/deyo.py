from math import log

import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/Jhyun17/DeYO
# Paper : https://openreview.net/forum?id=9w3iw8wDuE
# Note  : This version largely follows the code structure found in eta.py. The momentum parameter is only for BN layers.
#         Unlike the original, BN layers track running statistics (resetting source stats.) to enable frozen inference.
#         Patch-shuffle augmentation is implemented in pure PyTorch (no einops). Entropy and PLPD reweighting are always
#         applied. Only the 'patch' augmentation type is supported.
class DeYO(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1,
                 e1_threshold=0.5 * log(1000),  # Entropy filter: admits more samples before the PLPD stage culls further
                 e2_threshold=0.4 * log(1000),  # Entropy reweighting reference: same value as ETA/SAR
                 p_threshold=0.2,               # PLPD filter: retain samples whose prediction drops after patch shuffle
                 patch_len=4):                  # Number of patches per spatial dimension (must divide image resolution)
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.e1_threshold = e1_threshold
        self.e2_threshold = e2_threshold
        self.p_threshold = p_threshold
        self.patch_len = patch_len

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

                # 1. Entropy filter: keep low-entropy, confident samples
                mask_1 = entropy < self.e1_threshold
                if not mask_1.any():
                    self.optimizer.zero_grad()
                    return outputs.softmax(1), prediction_time

                entropy_filtered = entropy[mask_1]
                outputs_filtered = outputs[mask_1]

                # 2. Patch shuffle: perturb spatial structure to measure prediction sensitivity
                x_prime = self._patch_shuffle(x[mask_1].detach())
                with torch.no_grad():
                    outputs_prime = self.model(x_prime)

                prob_filtered = outputs_filtered.softmax(1)
                prob_prime = outputs_prime.softmax(1)
                cls1 = prob_filtered.argmax(dim=1)

                # PLPD: how much the predicted class probability drops after perturbation
                plpd = (torch.gather(prob_filtered, 1, cls1.unsqueeze(1)) -
                        torch.gather(prob_prime, 1, cls1.unsqueeze(1))).squeeze(1)

                # 3. PLPD filter: keep samples whose prediction was grounded in spatial structure
                mask_2 = plpd > self.p_threshold
                if not mask_2.any():
                    self.optimizer.zero_grad()
                    return outputs.softmax(1), prediction_time

                entropy_filtered = entropy_filtered[mask_2]
                plpd = plpd[mask_2]

                # 4. Reweighted entropy loss: upweight confident, spatially-grounded samples
                weight = (torch.exp(self.e2_threshold - entropy_filtered.detach()) +
                          torch.exp(plpd.detach()))
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

    def freeze(self):
        self.frozen = True
        self.model.eval()

    def unfreeze(self):
        self.frozen = False
        self.model.train()

    @staticmethod
    def collect_params(model):
        # Skip top layers for adaptation: layer4 for ResNets and blocks 9-11 for ViT-Base
        skip_in_name = ["layer4", "blocks.9", "blocks.10", "blocks.11", "norm."]
        skip_is_name = ["norm"]

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

    def _patch_shuffle(self, x):
        b, c, h, w = x.shape
        ph, pw = h // self.patch_len, w // self.patch_len

        # Rearrange into patches: (b, c, h, w) -> (b, patch_len^2, c, ph, pw)
        x = x.reshape(b, c, self.patch_len, ph, self.patch_len, pw)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, self.patch_len ** 2, c, ph, pw)

        # Shuffle patches independently per sample
        perm_idx = torch.argsort(torch.rand(b, self.patch_len ** 2, device=x.device), dim=-1)
        x = x[torch.arange(b, device=x.device).unsqueeze(-1), perm_idx]

        # Restore spatial layout: (b, patch_len^2, c, ph, pw) -> (b, c, h, w)
        x = x.reshape(b, self.patch_len, self.patch_len, c, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

        return x

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
        assert model.training, "DeYO requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "DeYO needs some parameters with gradients."
        assert not all(param_grads), "DeYO should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "DeYO requires normalization layers."
