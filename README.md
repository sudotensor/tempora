# Tempora

Current evaluations of test-time adaptation (TTA) methods overlook the accuracy-latency trade-off, despite its essential role in real-world deployment. This project evaluates TTA methods across three models of time-contingent utility to reveal how temporal pressures reorder method rankings:

1. **Discrete utility**: Hard deadlines where adaptation overhead causes the pipeline to miss incoming batches.
2. **Continuous utility**: Soft penalties where late predictions lose value according to a hyperbolic decay.
3. **Amortised utility**: Budgeted overhead where adaptation halts after exhausting a fixed compute budget.

These utility metrics reveal rank instability: the optimal method under offline evaluation frequently underperforms under temporal pressure. This codebase hosts the implementation of "Tempora: Time-Contingent Utility Metrics for Test-Time Adaptation" (ICML 2025 submission).

## Datasets

CIFAR-10/100 and ImageNet dataloaders with support for CIFAR-C and ImageNet-C corruptions (15 types, 5 severity levels). ImageNet validation data requires preprocessing with `valprep.sh` to organize images into class-wise folders.

## Models

ResNet-18, ResNet-50, MobileNet-V2, and MobileNet-V3-Small architectures with CIFAR-10/100 and ImageNet support. Models are patched to optionally return intermediate features for test-time adaptation. Pre-trained ImageNet-1K weights available; CIFAR models require custom checkpoints.

## Methods

Eight fully test-time adaptation methods that implement a unified API: AdaBN, PredBN, NEO, LAME, Tent, ETA, SHOT, and SAR. All methods were updated to support frozen inference where the forward pass is completely static.
