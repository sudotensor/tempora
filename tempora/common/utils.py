import json
import math
import random
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import numpy as np
import torch

from .constants import BLUR_CORRUPTIONS, CORRUPTIONS, DIGITAL_CORRUPTIONS, NOISE_CORRUPTIONS, WEATHER_CORRUPTIONS
from .datasets import get_cifar_dataloader, get_imagenet_dataloader
from .models import get_mobilenet, get_resnet


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


def parse_distributions(ds):
    out = []
    for d in ds:
        match d:
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
                out.append(d)
    return list(dict.fromkeys(out))  # Remove duplicates while preserving order


def print_arguments(args):
    spacer = max(len(k) for k in vars(args).keys())

    rows = ["-" * 32]
    for k, v in vars(args).items():
        rows.append(f"{k:<{spacer}} : {str(v) if v is not None else 'None'}")
    rows.append("-" * 32)

    print("\n".join(rows))


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


def setup_model(arch, dataset, ckpt, device):
    if arch in ["resnet18", "resnet50"]:
        return get_resnet(arch, dataset, ckpt, device)
    if arch in ["mobilenet_v2", "mobilenet_v3_small"]:
        return get_mobilenet(arch, dataset, ckpt, device)

    raise ValueError(f"Unknown architecture: {arch}")


def setup_method(method, model, dataset):
    # Avoids circular import with stopwatch function
    from .methods import ETA, LAME, NEO, SAR, SHOT, AdaBN, Basic, PredBN, Tent
    from .methods.utils import SAM

    num_classes = {"cifar-10": 10, "cifar-100": 100, "imagenet": 1000}[dataset]
    e_threshold = 0.4 * math.log(num_classes)
    d_threshold = {"cifar-10": 0.4, "cifar-100": 0.1, "imagenet": 0.05}[dataset]  # ~ sqrt(1/c)

    match method:
        # Gradient-free
        case "adabn":
            return AdaBN(model)
        case "basic":
            return Basic(model)
        case "lame":
            return LAME(model, neighbors=5, max_steps=100)
        case "neo":
            return NEO(model)
        case "predbn":
            return PredBN(model)
        # Gradient-based
        case "eta":
            ps, _ = ETA.collect_params(model)
            optim = torch.optim.SGD(ps, lr=0.00025, momentum=0.9)
            return ETA(model, optim, reforward=False, momentum=0.1, e_threshold=e_threshold, d_threshold=d_threshold)
        case "sar":
            ps, _ = SAR.collect_params(model)
            optim = SAM(ps, torch.optim.SGD, rho=0.05, adaptive=False, lr=0.00025, momentum=0.9)
            return SAR(model, optim, reforward=False, momentum=0.1, e_threshold=e_threshold)
        case "shot":
            ps, _ = SHOT.collect_params(model)
            optim = torch.optim.SGD(ps, lr=0.00025, momentum=0.9)
            return SHOT(model, optim, reforward=False)
        case "tent":
            ps, _ = Tent.collect_params(model)
            optim = torch.optim.SGD(ps, lr=0.00025, momentum=0.9)
            return Tent(model, optim, reforward=False, momentum=0.1)
        case _:
            raise ValueError(f"Unknown method: {method}")


def save_results(rs, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arguments = {k: str(v) for k, v in vars(args).items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = args.output_dir / f"{timestamp}.json"
    with open(filename, "w") as f:
        json.dump({"timestamp": timestamp, "arguments": arguments, "results": rs}, f, indent=2)

    print(f"Results saved to {filename}")