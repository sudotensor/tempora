import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from tempora.common.constants import DATASETS, DEVICES, METHODS
from tempora.common.recorders import OfflineRecorder
from tempora.common.utils import (
    parse_distributions,
    print_arguments,
    save_results,
    setup_dataloader,
    setup_determinism,
    setup_method,
    setup_model,
    stopwatch,
    synchronise,
)


@torch.no_grad()
def runner(method, dataloader, recorder, device):
    for i, (images, labels) in enumerate(tqdm(dataloader, leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        if i == 0:  # Warm-up
            for _ in range(5):
                with synchronise(device):
                    _ = method(images + torch.randn_like(images), device)
            method.reset()

        (outputs, prediction_time), wall_clock_time = stopwatch(device, lambda: method(images, device))
        
        num_correct = (outputs.argmax(1) == labels).sum().item()
        num_samples = labels.size(0)

        recorder.record(num_correct, num_samples, prediction_time, wall_clock_time)

    recorder.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core arguments
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    # Dataset arguments
    parser.add_argument("--dataset-name", choices=DATASETS, default="cifar-10")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--dataset-dist", nargs="+", default=["clean"])
    # Model arguments
    parser.add_argument("--model-arch", type=str, default="resnet50")
    parser.add_argument("--model-ckpt", type=Path, default=None)
    # Adaptation arguments
    parser.add_argument("--method", choices=METHODS, default="basic")

    args = parser.parse_args()
    print_arguments(args)

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    
    setup_determinism(args.seed, args.device)

    model = setup_model(args.model_arch, args.dataset_name, args.model_ckpt, args.device)
    method = setup_method(args.method, model, args.dataset_name, args.model_arch, args.device)

    rs = {}
    ds = parse_distributions(args.dataset_dist)
    for d in ds:
        setup_determinism(args.seed, args.device)

        recorder = OfflineRecorder()
        kwargs = {"batch_size": args.batch_size, "drop_last": True, "shuffle": True}
        dataloader = setup_dataloader(args.model_arch, args.dataset_name, args.dataset_root, d, **kwargs)

        print(d)
        runner(method, dataloader, recorder, args.device)
        rs[d] = recorder.emit()
        
        method.reset()
    
    save_results(rs, args)
