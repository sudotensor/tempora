import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet(arch: str, dataset: str, checkpoint: str | None = None, device: str = "cpu"):
    """Get a patched ResNet model for the specified architecture variant and dataset

    Args:
        arch:      'resnet18' or 'resnet50'
        dataset:   'cifar-10', 'cifar-100' or 'imagenet'
        ckpt_path: Path to custom weights file (optional for 'imagenet', required for 'cifar-10' and 'cifar-100')
        device:    Device to load model on
    """

    if arch not in ["resnet18", "resnet50"]:
        raise ValueError(f"Unsupported architecture: {arch}. Choose 'resnet18' or 'resnet50'")

    if dataset not in ["cifar-10", "cifar-100", "imagenet"]:
        raise ValueError(f"Unsupported dataset: {dataset}. Choose 'cifar-10', 'cifar-100' or 'imagenet'")

    kwargs = {}
    match dataset:
        case "cifar-10":
            kwargs["num_classes"] = 10
        case "cifar-100":
            kwargs["num_classes"] = 100
        case "imagenet":
            if checkpoint is None:
                kwargs["weights"] = "IMAGENET1K_V1"

    match arch:
        case "resnet18":
            model = models.resnet18(**kwargs)
        case "resnet50":
            model = models.resnet50(**kwargs)

    # Dataset-specific architecture modifications (following huyvnphan/PyTorch_CIFAR10)
    # Note: They don't set model.maxpool to nn.Identity() while training unlike other implementations.
    if dataset == "cifar-10" or dataset == "cifar-100":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    if checkpoint is None and (dataset == "cifar-10" or dataset == "cifar-100"):
        print("Warning: Checkpoint recommended for CIFAR-10/100 but none provided!")
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))

    model = _patch_resnet(model)
    model.to(device)
    model.eval()

    return model


def _patch_resnet(model):
    """Monkeypatch a ResNet model to return features along with predictions if requested"""

    def forward(self, x, return_feature_only=False, return_feature_and_logits=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        if return_feature_only:
            return x

        feature = x if return_feature_and_logits else None
        x = self.fc(x)

        if return_feature_and_logits:
            return x, feature

        return x

    # In torchvision, the forward method calls self._forward_impl to support torchscript (see link). We ignore this.
    # https://stackoverflow.com/q/72111540
    model.forward = forward.__get__(model, model.__class__)

    return model