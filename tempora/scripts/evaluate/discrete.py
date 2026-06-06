import argparse
from collections import deque
from pathlib import Path

import torch
from tqdm import tqdm

from tempora.common.constants import DATASETS, DEVICES, METHODS
from tempora.common.recorders import DiscreteRecorder
from tempora.common.utils import (
    get_class_mask,
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


# Discrete-time simulation with queuing where batches arrive every interval (ms). Arrivals enter a fixed-capacity queue
# when the pipeline is busy. If the queue is full, oldest queued batches are evicted and receive no predictions. When 
# free, the pipeline picks up the oldest queued batch. The queue keeps the pipeline continuously fed. The pipeline can
# only process one batch at a time. When the wall clock time exceeds the interval, the pipeline is bottlenecked.
# A queue size of 1 removes idle gaps between processing, improving availability. Larger queues only affect which 
# batches are processed, not how many; they only improve availability as a consequence of the drain. The effective
# response latency of a batch is its prediction time + queue wait time. 
@torch.no_grad()
def runner(method, dataloader, recorder, device, interval, queue_size=0, class_mask=None):
    def process_job(start_time, index, images, labels):  # Run adaptation on a batch (job)
        (outputs, prediction_time), wall_clock_time = stopwatch(device, lambda: method(images, device))
        queue_wait_time = start_time - (index * interval)

        if class_mask is not None:
            outputs = outputs[:, class_mask]

        num_correct = (outputs.argmax(1) == labels).sum().item()
        num_samples = labels.size(0)

        recorder.record(num_correct, num_samples, 0, start_time, prediction_time, wall_clock_time, queue_wait_time)
        return wall_clock_time

    queue = deque()

    arrive_time = 0.0
    finish_time = 0.0
    for i, (images, labels) in enumerate(tqdm(dataloader, leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        if i == 0:  # Warm-up
            for _ in range(5):
                with synchronise(device):
                    _ = method(images + torch.randn_like(images), device)
            method.reset()

        arrive_time = i * interval

        # 1. Pick up a new job while the pipeline is free
        while finish_time <= arrive_time and queue:
            finish_time += process_job(finish_time, *queue.popleft())

        if finish_time <= arrive_time:
            finish_time = arrive_time + process_job(arrive_time, i, images, labels)
            continue

        # 2. Handle new arrival when busy
        if queue_size == 0:
            recorder.record(0, labels.size(0), labels.size(0), 0, 0, 0, 0)
        elif len(queue) < queue_size:
            queue.append((i, images, labels))
        else:
            _, _, old_labels = queue.popleft()
            recorder.record(0, old_labels.size(0), old_labels.size(0), 0, 0, 0, 0)
            queue.append((i, images, labels))

    # 3. Drain the queue
    while queue:
        finish_time += process_job(finish_time, *queue.popleft())

    recorder.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Core arguments
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--device", choices=DEVICES, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--queue-size", type=int, default=0)
    parser.add_argument("--interval", type=float, default=50.0)
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

        recorder = DiscreteRecorder()
        kwargs = {"batch_size": args.batch_size, "drop_last": True, "shuffle": True}
        dataloader = setup_dataloader(args.model_arch, args.dataset_name, args.dataset_root, d, **kwargs)
        class_mask = get_class_mask(d, args.device)

        print(d)
        runner(method, dataloader, recorder, args.device, args.interval, args.queue_size, class_mask)
        rs[d] = recorder.emit()

        method.reset()
    
    save_results(rs, args)
