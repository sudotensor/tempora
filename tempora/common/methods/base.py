from abc import ABC, abstractmethod

import torch.nn as nn


class Method(nn.Module, ABC):
    """Abstract base class for test-time adaptation methods."""

    @abstractmethod
    def forward(self, x):
        pass

    # Reset to the initially configured state as if no forward calls were made.
    # It doesn't affect the chosen freeze/unfreeze mode.
    @abstractmethod
    def reset(self):
        pass

    # Pauses adaptation to enable frozen inference.
    @abstractmethod
    def freeze(self):
        pass

    # Resumes adaptation.
    @abstractmethod
    def unfreeze(self):
        pass
