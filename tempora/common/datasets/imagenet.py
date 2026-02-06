import os

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ..constants import CORRUPTIONS


def get_imagenet_dataloader(
    root: str,
    arch: str,
    variant: str = "clean",
    severity: int = 5,
    batch_size: int = 64,
    shuffle: bool = True,
    split: bool = False,
    resolution: int = 224,
    **kwargs,
):
    """Get an ImageNet validation dataloader with appropriate transforms

    Args:
        root:        Path to the directory to load the dataset from
        variant:     Corruption name or 'clean' (defaults to 'clean')
        severity:    Corruption severity 1-5 (defaults to 5, ignored for 'clean')
        batch_size:  Batch size (defaults to 32)
        shuffle:     Whether to shuffle data (defaults to True)
        split:       Create a 90/10
        resolution:  Image resolution for center crop (defaults to 224)
        **kwargs:    Additional arguments passed to DataLoader
    """

    normalize = None
    match arch:
        case "vit_base_patch16_224":
            # See https://huggingface.co/docs/transformers/v4.13.0/en/model_doc/vit#transformers.ViTFeatureExtractor
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        case _:
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if variant == "clean":
        transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])

        validation_path = os.path.join(root, "val")
        dataset = ImageFolder(validation_path, transform=transform)
    else:
        assert variant in CORRUPTIONS, f"Corruption {variant} not in {CORRUPTIONS}"
        assert 1 <= severity <= 5, "Severity must be 1-5"

        transform = transforms.Compose([transforms.CenterCrop(resolution), transforms.ToTensor(), normalize])

        corruption_path = os.path.join(root, variant, str(severity))
        dataset = ImageFolder(corruption_path, transform=transform)

    if not split:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    validation_dataset, test_dataset = random_split(dataset, [5000, len(dataset) - 5000])
    validation_loader = DataLoader(validation_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return validation_loader, test_loader
