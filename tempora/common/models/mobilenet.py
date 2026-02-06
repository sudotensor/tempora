import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import InvertedResidual


def get_mobilenet(arch: str, dataset: str, checkpoint: str | None = None, device: str = "cpu"):
    """Get a patched MobileNet model for the specified architecture variant and dataset

    Args:
        arch:      'mobilenet_v2' and 'mobilenet_v3_small'
        dataset:   'cifar-10', 'cifar-100' or 'imagenet'
        checkpoint: Path to custom weights file (optional for 'imagenet', required for 'cifar-10' and 'cifar-100')
        device:    Device to load model on
    """

    if arch not in ["mobilenet_v2", "mobilenet_v3_small"]:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'mobilenet_v2' or 'mobilenet_v3_small'")
    if dataset not in ["cifar-10", "cifar-100", "imagenet"]:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'cifar-10', 'cifar-100' or 'imagenet'")

    kwargs = {}
    match dataset:
        case "cifar-10":
            kwargs["num_classes"] = 10
        case "cifar-100":
            kwargs["num_classes"] = 100
        case "imagenet":
            kwargs["num_classes"] = 1000
            if checkpoint is None:
                kwargs["weights"] = "IMAGENET1K_V1"

    match arch:
        case "mobilenet_v2":
            model = models.mobilenet_v2(**kwargs)
        case "mobilenet_v3_small":
            model = models.mobilenet_v3_small(**kwargs)

    # Dataset-specific architecture modifications for CIFAR (following kuangliu/pytorch-cifar)
    if dataset == "cifar-10" or dataset == "cifar-100":
        model.features[0][0].stride = (1, 1)  # type:ignore
        model.features[2] = InvertedResidual(model.features[1].out_channels, model.features[2].out_channels, 1, 6)  # type:ignore

    if checkpoint is None and (dataset == "cifar-10" or dataset == "cifar-100"):
        print("Warning: Checkpoint recommended for CIFAR-10/100 but none provided!")
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))

    model = _patch_mobilenet(model)
    model.to(device)
    model.eval()

    return model


def _patch_mobilenet(model):
    """Monkeypatch a MobileNet model to return features along with predictions if requested"""

    def forward(self, x, return_feature_only=False, return_feature_and_logits=False):
        feature = None
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        if return_feature_only:
            return x

        feature = x if return_feature_and_logits else None
        x = self.classifier(x)

        if return_feature_and_logits:
            return x, feature

        return x

    # In torchvision, the forward method calls self._forward_impl to support torchscript
    model.forward = forward.__get__(model, model.__class__)

    return model
