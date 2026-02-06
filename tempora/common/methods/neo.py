import torch

from ..utils import stopwatch
from .base import Method


# Source: https://github.com/awesomealex1/NEO
# Paper : https://arxiv.org/abs/2510.05635
# Note  : NEO tracks the running mean of the batch features to center the feature distribution. This version uses an
#         algebraically equivalent formula for updating the feature mean.
class NEO(Method):
    def __init__(self, model):
        super().__init__()
        self.frozen = False

        self.model = self._configure_model(model)
        self.classifier = self._get_classifier(model)
        self.featurizer = self._make_featurizer(model)

        self.feature_mean = torch.zeros(self.classifier.weight.data.shape[1], device=self.classifier.weight.device)
        self.sample_count = 0

    def forward(self, x, device):
        @torch.no_grad()
        def _forward():
            if self.frozen:
                return self.model(x).softmax(1)

            features = self.featurizer(x)

            # Update running mean of features
            batch_size = features.size(0)
            batch_mean = features.mean(0)
            total_samples = self.sample_count + batch_size

            self.feature_mean = (self.sample_count * self.feature_mean + batch_size * batch_mean) / total_samples
            self.sample_count = total_samples

            outputs = self.classifier(features - self.feature_mean)  # Feature centering
            return outputs.softmax(1)

        return stopwatch(device, _forward)

    def reset(self):
        self.feature_mean.zero_()
        self.sample_count = 0

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def _configure_model(self, model):
        model.eval()
        model.requires_grad_(False)

        return model

    @staticmethod
    def _get_classifier(model):
        if hasattr(model, "fc"):
            return model.fc
        elif hasattr(model, "classifier"):
            return model.classifier
        elif hasattr(model, "head"):
            return model.head
        raise ValueError("Cannot find classifier layer")

    @staticmethod
    def _make_featurizer(model):
        return lambda x: model(x, return_feature_only=True)
