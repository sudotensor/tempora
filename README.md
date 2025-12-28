# Tempora: Characterising the time-contingent utility of fully online test-time adaptation

Current evaluations of test-time adaptation (TTA) methods often overlook the accuracy-latency trade-off, despite its essential role in real-world deployment. This project benchmarks TTA methods across three models of time-contingent utility to reveal how temporal pressures reorder method rankings:

1. Discrete: Binary utility based on hard deadlines where adaptation overhead causes the system to miss incoming batches, which receive null predictions.
2. Continuous: Decayed utility based on a hyperbolic penalty applied to delayed predictions, which simulates the diminishing value of slow responses in interactive applications.
3. Amortised: Pareto utility based on task performance across various temporal budgets, which leaves the trade-off open to interpretation.

## Datasets

CIFAR-10/100 and ImageNet dataloaders with support for CIFAR-C and ImageNet-C corruptions (15 types, 5 severity levels). ImageNet validation data requires preprocessing with `valprep.sh` to organize images into class-wise folders.
