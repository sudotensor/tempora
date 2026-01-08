import torch
import torch.nn as nn

from ..utils import stopwatch
from .base import Method
from .utils import get_cpu_snapshot


# Source: https://github.com/tim-learn/SHOT/blob/master/object/image_pretrained.py
# Paper : https://arxiv.org/abs/2002.08546
# Note  : This version implements SHOT-IM and largely follows the code structure found in tent.py. The core differences
#         are the loss function and parameterisation.
class SHOT(Method):
    def __init__(self, model, optimizer, reforward=False):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model)
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
                m_probs = outputs.softmax(1).mean(0)  # Marginal probabilities
                entropy = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)  # Conditional entropy
                diversity = (m_probs * torch.log(m_probs + torch.finfo(outputs.dtype).eps)).sum(-1)  # Marginal entropy

                loss = entropy.mean(0) + diversity  # Minimise conditional entropy, maximise marginal entropy
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
        ps = []
        ns = []

        # Collect all params. except ones from known classifiers; double check the param. group for new networks!
        # Transformer encoders have linear layers, so we can't do a general isinstance(m, nn.Linear) check.
        for np, p in model.named_parameters():
            if p.requires_grad and not any(x in np for x in ["fc", "classifier", "heads"]):
                ps.append(p)
                ns.append(np)

        return ps, ns

    def _configure_model(self, model):
        model.train()
        model.requires_grad_(True)

        for nm, m in model.named_modules():
            if any(x in nm for x in ["fc", "classifier", "heads"]):
                if isinstance(m, nn.Linear):
                    m.requires_grad_(False)

        return model

    def _check_model(self, model):
        assert model.training, "SHOT requires train mode."

        param_grads = [p.requires_grad for p in model.parameters()]
        assert any(param_grads), "SHOT needs some parameters with gradients."
