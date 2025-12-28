import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset

from ..constants import CORRUPTIONS


class CorruptedCIFARDataset(VisionDataset):
    """Generic CIFAR-C corrupted test dataset with memory mapping"""

    def __init__(self, root: str, corruption: str, severity: int = 5, transform=None):
        super().__init__(root, transform=transform)

        assert corruption in CORRUPTIONS, f"Corruption {corruption} not in {CORRUPTIONS}"
        assert 1 <= severity <= 5, "Severity must be 1-5"

        self.corruption = corruption
        self.severity = severity

        # Load data and labels with memory mapping
        data_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")

        self.data = np.load(data_path, mmap_mode="r")
        self.targets = np.load(labels_path, mmap_mode="r").astype(np.int64)

        # Compute the range of the requested severity subset
        self.start_idx = (severity - 1) * 10000
        self.end_idx = severity * 10000

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, index):
        # Access data with offset for severity level
        actual_idx = self.start_idx + index
        img = self.data[actual_idx]
        target = self.targets[actual_idx]

        # Convert to PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar_dataloader(
    name: str,
    root: str,
    variant: str = "clean",
    severity: int = 5,
    batch_size: int = 64,
    shuffle: bool = True,
    split: bool = False,
    **kwargs,
):
    """Get CIFAR test dataloader(s) with appropriate transforms

    Args:
        name:       'CIFAR10' or 'CIFAR100'
        root:       Path to the directory to load the dataset from
        variant:    Corruption name or 'clean' (defaults to 'clean')
        severity:   Corruption severity 1-5 (defaults to 5, ignored for 'clean')
        batch_size: Batch size (defaults to 64)
        shuffle:    Whether to shuffle data (defaults to True)
        split:      Create a 90/10 validation split from the test set (defaults to False)
        **kwargs:   Additional arguments passed to DataLoader

    Returns:
        DataLoader if split=False, tuple of (validation_loader, test_loader) if split=True
    """

    if name == "CIFAR10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        dataset_class = CIFAR10
    elif name == "CIFAR100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        dataset_class = CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {name}. Choose 'CIFAR10' or 'CIFAR100'")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if variant == "clean":
        dataset = dataset_class(root, train=False, transform=transform)
    else:
        dataset = CorruptedCIFARDataset(root, variant, severity, transform=transform)

    kwargs.update({"batch_size": batch_size, "shuffle": shuffle})

    if not split:
        return DataLoader(dataset, **kwargs)

    validation_dataset, test_dataset = random_split(dataset, [1000, len(dataset) - 1000])
    return DataLoader(validation_dataset, batch_size=batch_size), DataLoader(test_dataset, **kwargs)
