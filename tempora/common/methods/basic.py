from ..utils import stopwatch
from .base import Method


# Basic wrapper for the no adaptation baseline
class Basic(Method):
    def __init__(self, model):
        super().__init__()
        self.model = self._configure_model(model)

    def forward(self, x, device):
        return stopwatch(device, lambda: self.model(x).softmax(1))

    def reset(self):
        pass

    def freeze(self):
        pass

    def unfreeze(self):
        pass

    def _configure_model(self, model):
        model.eval()
        model.requires_grad_(False)

        return model
