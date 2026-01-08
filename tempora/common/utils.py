import random
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

from .constants import BLUR_CORRUPTIONS, CORRUPTIONS, DIGITAL_CORRUPTIONS, NOISE_CORRUPTIONS, WEATHER_CORRUPTIONS
from .datasets import get_cifar_dataloader, get_imagenet_dataloader


@contextmanager
def synchronise(device) -> Iterator[None]:
    if device == "cuda": torch.cuda.synchronize()  # noqa: E701
    yield
    if device == "cuda": torch.cuda.synchronize()  # noqa: E701


def stopwatch(device, func):
    with synchronise(device):
        start = time.perf_counter()
        result = func()

    return result, (time.perf_counter() - start) * 1000


def parse_corruptions(cs):
    out = []
    for c in cs:
        match c:
            case "all":
                out.extend(["clean"] + CORRUPTIONS)
            case "noise":
                out.extend(NOISE_CORRUPTIONS)
            case "blur":
                out.extend(BLUR_CORRUPTIONS)
            case "weather":
                out.extend(WEATHER_CORRUPTIONS)
            case "digital":
                out.extend(DIGITAL_CORRUPTIONS)
            case _:
                out.append(c)
    return list(dict.fromkeys(out))  # Remove duplicates while preserving order


def setup_determinism(seed, device):
    # the following function also sets:
    # 1. torch.backends.cudnn.deterministic
    # 2. torch.utils.deterministic.full_uninitialized_memory
    torch.use_deterministic_algorithms(device != "cuda")
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_dataloader(dataset, root, variant, **kwargs):
    root = str(root / (dataset if variant == "clean" else f"{dataset}-c"))
    match dataset:
        case "cifar-10":
            return get_cifar_dataloader("CIFAR10", root, variant=variant, **kwargs)
        case "cifar-100":
            return get_cifar_dataloader("CIFAR100", root, variant=variant, **kwargs)
        case "imagenet":
            return get_imagenet_dataloader(root, variant=variant, **kwargs)

    raise ValueError(f"Unknown dataset: {dataset}")