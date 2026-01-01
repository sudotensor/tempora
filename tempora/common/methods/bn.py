import torch.nn as nn

from .base import Method


# Paper: https://openreview.net/forum?id=Hk6dkJQFx
# Note : We use AdaBN in an online manner instead of collecting the EMA stats. of the entire target domain before eval.
class AdaBN(Method):
    def __init__(self, model, momentum=0.1):
        super().__init__()
        self.bn_modules = []

        self.model = self._configure_model(model, momentum)

    def forward(self, x):
        return self.model(x).softmax(1)

    def reset(self):
        for _, m in self.bn_modules:
            m.reset_running_stats()

    def freeze(self):
        for _, m in self.bn_modules:
            m.training = False

    def unfreeze(self):
        for _, m in self.bn_modules:
            m.training = True

    def _configure_model(self, model, momentum):
        model.eval()
        model.requires_grad_(False)

        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = True
                m.momentum = momentum  # Defaults to 0.1
                m.reset_running_stats()  # Discard the source statistics
                self.bn_modules.append((nm, m))

        return model


# Paper: https://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-055.pdf
# Note : PredBN is a stateless adaptation method that discards the running stats. buffers.
#        It's near-identical to AdaBN when adapting but does not inherently submit a truly frozen model.
#        The source stats. must be restored for frozen inference.
class PredBN(Method):
    def __init__(self, model):
        super().__init__()
        self.bn_modules = []

        self.source_stats = self._get_stats(model)
        self.model = self._configure_model(model)

    def forward(self, x):
        return self.model(x).softmax(1)

    def reset(self):
        pass

    def freeze(self):
        # Buffer stats. only used when m.training = False and m.track_running_stats = True.
        for nm, m in self.bn_modules:
            m.training = False
            m.track_running_stats = True

            m.running_mean = self.source_stats[nm]["mean"]
            m.running_var = self.source_stats[nm]["var"]

    def unfreeze(self):
        for _, m in self.bn_modules:
            m.training = True
            m.track_running_stats = False

            m.running_mean = None
            m.running_var = None

    def _configure_model(self, model):
        model.eval()
        model.requires_grad_(False)

        # Discard source statistics and instead use statistics from the current mini-batch.
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = True
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
                self.bn_modules.append((nm, m))

        return model

    def _get_stats(self, model):
        stats = {}
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                stats[nm] = {
                    "mean": m.running_mean.clone() if m.running_mean is not None else None,
                    "var": m.running_var.clone() if m.running_var is not None else None,
                }

        return stats
