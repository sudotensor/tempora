import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Rebuttal ablation: SHOT-IM with norm-layer-only parameterisation (BN/GN/LN weight+bias),
# matching ETA's collect_params and _configure_model. The IM loss is unchanged.
# Purpose: isolate whether SHOT-IM's amortised resilience comes from the loss or the parameterisation.
class SHOTNorm(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        def _forward():
            outputs, prediction_time = None, 0.0

            with torch.enable_grad():
                outputs, prediction_time = stopwatch(device, lambda: self.model(x))
                m_probs = outputs.softmax(1).mean(0)
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
                diversity = (m_probs * torch.log(m_probs + torch.finfo(outputs.dtype).eps)).sum(-1)

                loss = entropy.mean(0) + diversity
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
                m.reset_running_stats()
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model

    def _check_model(self, model):
        assert model.training, "SHOTNorm requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "SHOTNorm needs some parameters with gradients."
        assert not all(param_grads), "SHOTNorm should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "SHOTNorm requires normalization layers."
