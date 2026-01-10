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


# Amortised evaluation with a total adaptation overhead budget. The method adapts normally until its cumulative overhead
# exceeds the budget, at which point it is frozen for the remainder of the evaluation. This measures how effectively a
# method uses a fixed time budget. Efficient methods will adapt more batches before exhausting the budget.
@torch.no_grad()
def runner(method, dataloader, recorder, device, response_budget, overhead_budget):
    num_batches_cleared = 0
    is_budget_exhausted = False
    cumulative_overhead = 0.0
    for i, (images, labels) in enumerate(tqdm(dataloader, leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        if i == 0:  # Warm-up
            for _ in range(5):
                with synchronise(device):
                    _ = method(images + torch.randn_like(images), device)
            method.reset()

        if cumulative_overhead >= overhead_budget and not is_budget_exhausted:
            is_budget_exhausted = True
            method.freeze()

        (outputs, prediction_time), wall_clock_time = stopwatch(device, lambda: method(images, device))

        num_correct = (outputs.argmax(1) == labels).sum().item()
        num_samples = labels.size(0)

        if not is_budget_exhausted:
            cumulative_overhead += max(0.0, wall_clock_time - response_budget)
            num_batches_cleared += 1

        recorder.record(num_correct, num_samples, prediction_time, wall_clock_time)

    recorder.print()

    print(f"Batches Cleared : {num_batches_cleared}")
    print(f"Cumulative      : {cumulative_overhead:,.2f} ms")
    print(f"Remaining       : {max(0, overhead_budget - cumulative_overhead):,.2f} ms")
    print("-" * 32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core arguments
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--response-budget", type=float, default=50.0)
    parser.add_argument("--overhead-budget", type=float, default=1000.0)
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
    method = setup_method(args.method, model, args.dataset_name)

    rs = {}
    ds = parse_distributions(args.dataset_dist)
    for d in ds:
        setup_determinism(args.seed, args.device)

        recorder = OfflineRecorder()
        kwargs = {"batch_size": args.batch_size, "drop_last": True, "shuffle": True}
        dataloader = setup_dataloader(args.dataset_name, args.dataset_root, d, **kwargs)

        print(d)
        runner(method, dataloader, recorder, args.device, args.response_budget, args.overhead_budget)
        rs[d] = recorder.emit()

        method.reset()
        method.unfreeze()

    save_results(rs, args)
