import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/DequanWang/tent/blob/master/tent.py
# Paper : https://openreview.net/forum?id=uXl3bZLkr3c
# Note  : This version differs from the original in three ways: (1) it tracks running stats. to enable frozen inference,
#         (2) it drops batch-wise episodic resetting and multi-step gradient updates, and (3) it allows prediction after
#         adaptation via reforward. It's identical to the original during adaptation (for predict-then-adapt). The 
#         momentum parameter is only for BN layers.
class Tent(Method):
    def __init__(self, model, optimizer, reforward=False, momentum=0.1):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model, momentum)
        self.optimizer = optimizer
        self.reforward = reforward

        # Save the initial weights and optimiser states
        # Note: This might not work as intended for nested optimisers
        self.model_state = get_cpu_snapshot(self.model.state_dict())
        self.optimizer_state = get_cpu_snapshot(self.optimizer.state_dict())

        self._check_model(self.model)

    def forward(self, x, device):
        if self.frozen:
            with torch.no_grad():
                return stopwatch(device, lambda: self.model(x).softmax(1))

        outputs, prediction_time = None, 0.0
        with torch.enable_grad():
            outputs, prediction_time = stopwatch(device, lambda: self.model(x))
            entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)  # Conditional entropy

            loss = entropy.mean(0)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.reforward:
            with torch.no_grad():
                self.freeze()
                outputs, reforward_time = stopwatch(device, lambda: (self.freeze(), self.model(x), self.unfreeze())[1])
                self.unfreeze()
                prediction_time += reforward_time

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
                m.reset_running_stats()  # Discard source statistics
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)

        return model
    
    def _check_model(self, model):
        assert model.training, "TENT requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "TENT needs some parameters with gradients."
        assert not all(param_grads), "TENT should not update all parameters."

        has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)) for m in model.modules())
        assert has_norm, "TENT requires normalization layers."
