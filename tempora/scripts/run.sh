#!/usr/bin/env bash
# Experiment runner file for "Tempora: Characterising the time-contingent utility of online test-time adaptation" (ICML 2026)


# Instructions:
# 1. Please uncomment to run the evaluations.
# 2. Run this file from the project root directory.
#    Alternatively, prefix your PYTHONPATH with the project root and modify the evaluation blocks with the filepath:
#    PYTHONPATH=<path to root>:PYTHONPATH uv run python <path to file>.py [options]
# 3. Adjust the standard inference latency, thresholds, and budgets according to measurements on your hardware platform
#    Use the offline script for "basic" first and compute the mean and std. dev. of latencies across all distributions
#    Guideline: standard inference (reference) latency = mean + 6 * std. dev.


# 1. Offline evaluation
#    1.1 ResNet-50, ImageNet-C
#    1.2 ResNet-18, ImageNet-C
#    1.3 ViT-Base-Patch16-224, ImageNet-C
#    1.4 ResNet-18, CIFAR-10-C
#    1.5 ResNet-50, ImageNet-R
#    1.6 ResNet-50, ImageNet-V2
#    1.7 ViT-Base-Patch16-224, ImageNet-R
#    1.8 ViT-Base-Patch16-224, ImageNet-V2
#
# 1.1 ResNet-50, ImageNet-C
# Reference latency: 39.9 ms
# -------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method basic --note "Offline, Basic, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method adabn --note "Offline, AdaBN, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method lame  --note "Offline, LAME,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method neo   --note "Offline, NEO,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method tent  --note "Offline, Tent,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method eta   --note "Offline, ETA,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method shot  --note "Offline, SHOT,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method sar   --note "Offline, SAR,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method cmf   --note "Offline, CMF,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method deyo  --note "Offline, DeYO,  RN-50, IN-C"
#
# 1.2 ResNet-18, ImageNet-C
# Reference latency: 12.3 ms
# -------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method basic --note "Offline, Basic, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method adabn --note "Offline, AdaBN, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method lame  --note "Offline, LAME,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method neo   --note "Offline, NEO,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method tent  --note "Offline, Tent,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method eta   --note "Offline, ETA,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method shot  --note "Offline, SHOT,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/offline --model-arch resnet18 --method sar   --note "Offline, SAR,   RN-18, IN-C"
#
# 1.3 ViT-Base-Patch16-224, ImageNet-C
# Reference latency: 105.3 ms
# ------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method basic    --note "Offline, Basic,    ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method lame     --note "Offline, LAME,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method neo      --note "Offline, NEO,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method tent     --note "Offline, Tent,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method eta      --note "Offline, ETA,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method shot     --note "Offline, SHOT,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method sar      --note "Offline, SAR,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method spa      --note "Offline, SPA,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method deyo     --note "Offline, DeYO,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method cmf      --note "Offline, CMF,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method zerosiam --note "Offline, ZeroSIAM, ViT-Base-Patch16-224, IN-C"
#
# 1.4 ResNet-18, CIFAR-10-C
# Reference latency: 1.5 ms (GPU) vs 486.6 ms (RPi 5)
# Note: Weights checkpoint required at artefacts/weights/RN18-C10.pt
# -------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method basic  --note "Offline, Basic,   RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method adabn  --note "Offline, AdaBN,   RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method lame   --note "Offline, LAME,    RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method neo    --note "Offline, NEO,     RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method tent   --note "Offline, Tent,    RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method eta    --note "Offline, ETA,     RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method shot   --note "Offline, SHOT-IM, RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/offline --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method sar    --note "Offline, SAR,     RN-18, C-10"
#
# 1.5 ResNet-50, ImageNet-R
# --------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method basic --note "Offline, Basic, RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method adabn --note "Offline, AdaBN, RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method lame  --note "Offline, LAME,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method neo   --note "Offline, NEO,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method tent  --note "Offline, Tent,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method eta   --note "Offline, ETA,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method shot  --note "Offline, SHOT,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method sar   --note "Offline, SAR,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method cmf   --note "Offline, CMF,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/offline --model-arch resnet50 --method deyo  --note "Offline, DeYO,  RN-50, IN-R"
#
# 1.6 ResNet-50, ImageNet-V2
# --------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method basic --note "Offline, Basic, RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method adabn --note "Offline, AdaBN, RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method lame  --note "Offline, LAME,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method neo   --note "Offline, NEO,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method tent  --note "Offline, Tent,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method eta   --note "Offline, ETA,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method shot  --note "Offline, SHOT,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method sar   --note "Offline, SAR,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method cmf   --note "Offline, CMF,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/offline --model-arch resnet50 --method deyo  --note "Offline, DeYO,  RN-50, IN-V2"
#
# 1.7 ViT-Base-Patch16-224, ImageNet-R
# ------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method basic    --note "Offline, Basic,    ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method lame     --note "Offline, LAME,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method neo      --note "Offline, NEO,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method tent     --note "Offline, Tent,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method eta      --note "Offline, ETA,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method shot     --note "Offline, SHOT,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method sar      --note "Offline, SAR,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method spa      --note "Offline, SPA,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method cmf      --note "Offline, CMF,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method deyo     --note "Offline, DeYO,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method zerosiam --note "Offline, ZeroSIAM, ViT-Base-Patch16-224, IN-R"
#
# 1.8 ViT-Base-Patch16-224, ImageNet-V2
# -------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method basic    --note "Offline, Basic,    ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method lame     --note "Offline, LAME,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method neo      --note "Offline, NEO,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method tent     --note "Offline, Tent,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method eta      --note "Offline, ETA,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method shot     --note "Offline, SHOT,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method sar      --note "Offline, SAR,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method spa      --note "Offline, SPA,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method cmf      --note "Offline, CMF,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method deyo     --note "Offline, DeYO,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method zerosiam --note "Offline, ZeroSIAM, ViT-Base-Patch16-224, IN-V2"


# 2. Discrete evaluation
#    2.1  Unbuffered (Alfarra et al.), ResNet-50, ImageNet-C
#    2.2  Buffered, ResNet-50, ImageNet-C at various utilisation levels
#    2.3  Unbuffered (Alfarra et al.), ResNet-18, ImageNet-C
#    2.4  Buffered, ResNet-18, ImageNet-C at 100% utilisation only
#    2.5  Unbuffered (Alfarra et al.), ViT-B-Patch16-224, ImageNet-C
#    2.6  Buffered, ViT-Base-Patch16-224, ImageNet-C at various utilisation levels
#    2.7  Unbuffered (Alfarra et al.), ResNet-18, CIFAR-10-C
#    2.8  Buffered, ResNet-18, CIFAR-10-C at various utilisation levels
#    2.9  Unbuffered (Alfarra et al.), ResNet-50, ImageNet-R
#    2.10 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-V2
#    2.11 Buffered, ResNet-50, ImageNet-R at various utilisation levels
#    2.12 Buffered, ResNet-50, ImageNet-V2 at various utilisation levels
#    2.13 Unbuffered (Alfarra et al.), ViT-Base-Patch16-224, ImageNet-R
#    2.14 Unbuffered (Alfarra et al.), ViT-Base-Patch16-224, ImageNet-V2
#    2.15 Buffered, ViT-Base-Patch16-224, ImageNet-R at various utilisation levels
#    2.16 Buffered, ViT-Base-Patch16-224, ImageNet-V2 at various utilisation levels
#
# 2.1 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-C
# ------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method basic --note "Discrete, No queue, 39.9, Basic, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method adabn --note "Discrete, No queue, 39.9, AdaBN, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method lame  --note "Discrete, No queue, 39.9, LAME,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method neo   --note "Discrete, No queue, 39.9, NEO,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method tent  --note "Discrete, No queue, 39.9, Tent,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method eta   --note "Discrete, No queue, 39.9, ETA,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method shot  --note "Discrete, No queue, 39.9, SHOT,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method sar   --note "Discrete, No queue, 39.9, SAR,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method cmf   --note "Discrete, No queue, 39.9, CMF,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method deyo  --note "Discrete, No queue, 39.9, DeYO,  RN-50, IN-C"
#
# 2.2 Buffered, ResNet-50, ImageNet-C at various utilisation levels
# iat = {1, sqrt(2), 2, 2 * sqrt(2), 4} * 39.9 ms; this corresponds to rho in {100%, 70%, 50%, 35%, 25%} utilisation
# -----------------------------------------------------------------
# for info in "39.9:100" "56.4:70" "79.8:50" "112.8:35" "159.6:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic adabn lame neo tent eta shot sar deyo cmf; do
#         uv run python -m tempora.scripts.evaluate.discrete                  \
#             --dataset-name imagenet                                         \
#             --dataset-dist noise blur weather digital                       \
#             --output-dir output/imagenet-c/resnet-50/discrete/buffered-$rho \
#             --queue-size 1                                                  \
#             --interval $iat                                                 \
#             --model-arch resnet50                                           \
#             --method $method                                                \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, RN-50, IN-C"
#     done
# done
#
# 2.3 Unbuffered (Alfarra et al.), ResNet-18, ImageNet-C
# ------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method basic --note "Discrete, No queue, 12.3, Basic, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method adabn --note "Discrete, No queue, 12.3, AdaBN, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method lame  --note "Discrete, No queue, 12.3, LAME,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method neo   --note "Discrete, No queue, 12.3, NEO,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method tent  --note "Discrete, No queue, 12.3, Tent,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method eta   --note "Discrete, No queue, 12.3, ETA,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method shot  --note "Discrete, No queue, 12.3, SHOT,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 12.3 --model-arch resnet18 --method sar   --note "Discrete, No queue, 12.3, SAR,   RN-18, IN-C"
#
# 2.4 Buffered, ResNet-18, ImageNet-C at 100% utilisation only
# ------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method basic --note "Discrete, Single-slot queue, 12.3, Basic, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method adabn --note "Discrete, Single-slot queue, 12.3, AdaBN, RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method lame  --note "Discrete, Single-slot queue, 12.3, LAME,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method neo   --note "Discrete, Single-slot queue, 12.3, NEO,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method tent  --note "Discrete, Single-slot queue, 12.3, Tent,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method eta   --note "Discrete, Single-slot queue, 12.3, ETA,   RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method shot  --note "Discrete, Single-slot queue, 12.3, SHOT,  RN-18, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-18/discrete/buffered --queue-size 1 --interval 12.3 --model-arch resnet18 --method sar   --note "Discrete, Single-slot queue, 12.3, SAR,   RN-18, IN-C"
#
# 2.5 Unbuffered (Alfarra et al.), ViT-B-Patch16-224, ImageNet-C
# --------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method basic    --note "Discrete, No queue, 105.3, Basic,    ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method lame     --note "Discrete, No queue, 105.3, LAME,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method neo      --note "Discrete, No queue, 105.3, NEO,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method tent     --note "Discrete, No queue, 105.3, Tent,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method eta      --note "Discrete, No queue, 105.3, ETA,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method shot     --note "Discrete, No queue, 105.3, SHOT,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method sar      --note "Discrete, No queue, 105.3, SAR,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method spa      --note "Discrete, No queue, 105.3, SPA,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method deyo     --note "Discrete, No queue, 105.3, DeYO,     ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method cmf      --note "Discrete, No queue, 105.3, CMF,      ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method zerosiam --note "Discrete, No queue, 105.3, ZeroSIAM, ViT-Base-Patch16-224, IN-C"
#
# 2.6 Buffered, ViT-Base-Patch16-224, ImageNet-C at various utilisation levels
# iat = {1, sqrt(2), 2, 2 * sqrt(2), 4} * 105.3 ms; this corresponds to rho in {100%, 70%, 50%, 35%, 25%} utilisation
# ----------------------------------------------------------------------------
# for info in "105.3:100" "148.9:70" "210.6:50" "297.8:35" "421.2:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic lame neo tent eta shot sar spa deyo cmf zerosiam; do
#         uv run python -m tempora.scripts.evaluate.discrete                             \
#             --dataset-name imagenet                                                    \
#             --dataset-dist noise blur weather digital                                  \
#             --output-dir output/imagenet-c/vit-base-patch16-224/discrete/buffered-$rho \
#             --queue-size 1                                                             \
#             --interval $iat                                                            \
#             --model-arch vit_base_patch16_224                                          \
#             --method $method                                                           \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, ViT-Base-Patch16-224, IN-C"
#     done
# done
#
# 2.7 Unbuffered (Alfarra et al.), ResNet-18, CIFAR-10-C
# ------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method basic  --note "Discrete, No queue, 1.5, Basic,   RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method adabn  --note "Discrete, No queue, 1.5, AdaBN,   RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method lame   --note "Discrete, No queue, 1.5, LAME,    RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method neo    --note "Discrete, No queue, 1.5, NEO,     RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method tent   --note "Discrete, No queue, 1.5, Tent,    RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method eta    --note "Discrete, No queue, 1.5, ETA,     RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method shot   --note "Discrete, No queue, 1.5, SHOT-IM, RN-18, C-10"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name cifar-10 --dataset-dist noise blur weather digital --output-dir output/cifar-10-c/resnet-18/discrete/unbuffered --queue-size 0 --interval 1.5 --model-arch resnet18 --model-ckpt artefacts/weights/RN18-C10.pt --method sar    --note "Discrete, No queue, 1.5, SAR,     RN-18, C-10"
#
# 2.8 Buffered, ResNet-18, CIFAR-10-C at various utilisation levels
# iat = {1, sqrt(2), 2, 2*sqrt(2), 4} * 1.5 ms → rho in {100%, 70%, 50%, 35%, 25%}
# -----------------------------------------------------------------
# for info in "1.5:100" "2.1:70" "3.0:50" "4.2:35" "6.0:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic adabn lame neo tent eta shot sar; do
#         uv run python -m tempora.scripts.evaluate.discrete                  \
#             --dataset-name cifar-10                                         \
#             --dataset-dist noise blur weather digital                       \
#             --output-dir output/cifar-10-c/resnet-18/discrete/buffered-$rho \
#             --queue-size 1                                                  \
#             --interval $iat                                                 \
#             --model-arch resnet18                                           \
#             --model-ckpt artefacts/weights/RN18-C10.pt                      \
#             --method $method                                                \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, RN-18, C-10"
#         sleep 2
#     done
# done
#
# 2.9 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-R
# ------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method basic --note "Discrete, No queue, 39.9, Basic, RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method adabn --note "Discrete, No queue, 39.9, AdaBN, RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method lame  --note "Discrete, No queue, 39.9, LAME,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method neo   --note "Discrete, No queue, 39.9, NEO,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method tent  --note "Discrete, No queue, 39.9, Tent,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method eta   --note "Discrete, No queue, 39.9, ETA,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method shot  --note "Discrete, No queue, 39.9, SHOT,  RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method sar   --note "Discrete, No queue, 39.9, SAR,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method cmf   --note "Discrete, No queue, 39.9, CMF,   RN-50, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method deyo  --note "Discrete, No queue, 39.9, DeYO,  RN-50, IN-R"
#
# 2.10 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-V2
# --------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method basic --note "Discrete, No queue, 39.9, Basic, RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method adabn --note "Discrete, No queue, 39.9, AdaBN, RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method lame  --note "Discrete, No queue, 39.9, LAME,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method neo   --note "Discrete, No queue, 39.9, NEO,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method tent  --note "Discrete, No queue, 39.9, Tent,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method eta   --note "Discrete, No queue, 39.9, ETA,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method shot  --note "Discrete, No queue, 39.9, SHOT,  RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method sar   --note "Discrete, No queue, 39.9, SAR,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method cmf   --note "Discrete, No queue, 39.9, CMF,   RN-50, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --model-arch resnet50 --method deyo  --note "Discrete, No queue, 39.9, DeYO,  RN-50, IN-V2"
#
# 2.11 Buffered, ResNet-50, ImageNet-R at various utilisation levels
# iat = {1, sqrt(2), 2, 2*sqrt(2), 4} * 39.9 ms; rho in {100%, 70%, 50%, 35%, 25%}
# ------------------------------------------------------------------
# for info in "39.9:100" "56.4:70" "79.8:50" "112.8:35" "159.6:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic adabn lame neo tent eta shot sar cmf deyo; do
#         uv run python -m tempora.scripts.evaluate.discrete                  \
#             --dataset-name imagenet                                         \
#             --dataset-dist imagenet-r                                       \
#             --output-dir output/imagenet-r/resnet-50/discrete/buffered-$rho \
#             --queue-size 1                                                  \
#             --interval $iat                                                 \
#             --model-arch resnet50                                           \
#             --method $method                                                \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, RN-50, IN-R"; sleep 2
#     done
# done
#
# 2.12 Buffered, ResNet-50, ImageNet-V2 at various utilisation levels
# iat = {1, sqrt(2), 2, 2*sqrt(2), 4} * 39.9 ms; rho in {100%, 70%, 50%, 35%, 25%}
# -------------------------------------------------------------------
# for info in "39.9:100" "56.4:70" "79.8:50" "112.8:35" "159.6:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic adabn lame neo tent eta shot sar cmf deyo; do
#         uv run python -m tempora.scripts.evaluate.discrete                   \
#             --dataset-name imagenet                                          \
#             --dataset-dist imagenet-v2                                       \
#             --output-dir output/imagenet-v2/resnet-50/discrete/buffered-$rho \
#             --queue-size 1                                                   \
#             --interval $iat                                                  \
#             --model-arch resnet50                                            \
#             --method $method                                                 \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, RN-50, IN-V2"; sleep 2
#     done
# done
#
# 2.13 Unbuffered (Alfarra et al.), ViT-Base-Patch16-224, ImageNet-R
# ------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method basic    --note "Discrete, No queue, 105.3, Basic,    ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method lame     --note "Discrete, No queue, 105.3, LAME,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method neo      --note "Discrete, No queue, 105.3, NEO,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method tent     --note "Discrete, No queue, 105.3, Tent,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method eta      --note "Discrete, No queue, 105.3, ETA,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method shot     --note "Discrete, No queue, 105.3, SHOT,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method sar      --note "Discrete, No queue, 105.3, SAR,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method spa      --note "Discrete, No queue, 105.3, SPA,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method cmf      --note "Discrete, No queue, 105.3, CMF,      ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method deyo     --note "Discrete, No queue, 105.3, DeYO,     ViT-Base-Patch16-224, IN-R"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-r --output-dir output/imagenet-r/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method zerosiam --note "Discrete, No queue, 105.3, ZeroSIAM, ViT-Base-Patch16-224, IN-R"
#
# 2.14 Unbuffered (Alfarra et al.), ViT-Base-Patch16-224, ImageNet-V2
# -------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method basic    --note "Discrete, No queue, 105.3, Basic,    ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method lame     --note "Discrete, No queue, 105.3, LAME,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method neo      --note "Discrete, No queue, 105.3, NEO,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method tent     --note "Discrete, No queue, 105.3, Tent,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method eta      --note "Discrete, No queue, 105.3, ETA,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method shot     --note "Discrete, No queue, 105.3, SHOT,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method sar      --note "Discrete, No queue, 105.3, SAR,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method spa      --note "Discrete, No queue, 105.3, SPA,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method cmf      --note "Discrete, No queue, 105.3, CMF,      ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method deyo     --note "Discrete, No queue, 105.3, DeYO,     ViT-Base-Patch16-224, IN-V2"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist imagenet-v2 --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method zerosiam --note "Discrete, No queue, 105.3, ZeroSIAM, ViT-Base-Patch16-224, IN-V2"
#
# 2.15 Buffered, ViT-Base-Patch16-224, ImageNet-R at various utilisation levels
# iat = {1, sqrt(2), 2, 2*sqrt(2), 4} * 105.3 ms; rho in {100%, 70%, 50%, 35%, 25%}
# -----------------------------------------------------------------------------
# for info in "105.3:100" "148.9:70" "210.6:50" "297.8:35" "421.2:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic lame neo tent eta shot sar spa cmf deyo zerosiam; do
#         uv run python -m tempora.scripts.evaluate.discrete                             \
#             --dataset-name imagenet                                                    \
#             --dataset-dist imagenet-r                                                  \
#             --output-dir output/imagenet-r/vit-base-patch16-224/discrete/buffered-$rho \
#             --queue-size 1                                                             \
#             --interval $iat                                                            \
#             --model-arch vit_base_patch16_224                                          \
#             --method $method                                                           \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, ViT-Base-Patch16-224, IN-R"; sleep 2
#     done
# done
#
# 2.16 Buffered, ViT-Base-Patch16-224, ImageNet-V2 at various utilisation levels
# iat = {1, sqrt(2), 2, 2*sqrt(2), 4} * 105.3 ms; rho in {100%, 70%, 50%, 35%, 25%}
# ------------------------------------------------------------------------------
# for info in "105.3:100" "148.9:70" "210.6:50" "297.8:35" "421.2:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic lame neo tent eta shot sar spa cmf deyo zerosiam; do
#         uv run python -m tempora.scripts.evaluate.discrete                              \
#             --dataset-name imagenet                                                     \
#             --dataset-dist imagenet-v2                                                  \
#             --output-dir output/imagenet-v2/vit-base-patch16-224/discrete/buffered-$rho \
#             --queue-size 1                                                              \
#             --interval $iat                                                             \
#             --model-arch vit_base_patch16_224                                           \
#             --method $method                                                            \
#             --note "Discrete, ρ=$rho%, IAT=$iat, $method, ViT-Base-Patch16-224, IN-V2"; sleep 2
#     done
# done


# 3. Continuous evaluation
#    3.1 ResNet-50, ImageNet-C
#    3.2 ResNet-18, ImageNet-C
#    3.3 ViT-Base-Patch16-224, ImageNet-C
#    3.4 ResNet-50, ImageNet-R
#    3.5 ResNet-50, ImageNet-V2
#    3.6 ViT-Base-Patch16-224, ImageNet-R
#    3.7 ViT-Base-Patch16-224, ImageNet-V2
#
# Note: You will have to run offline evaluation and modify the input file paths accordingly.
#
# 3.1 ResNet-50, ImageNet-C
# -------------------------
# for threshold in 50 100 200 400 1000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_003617.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_005231.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_010828.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_012404.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_015129.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_021856.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_025057.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260111_033723.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260326_201551.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-50/offline/20260326_204931.json --output-dir output/imagenet-c/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 1
# done
#
# 3.2 ResNet-18, ImageNet-C
# -------------------------
# for threshold in 16 33 50 100 200 400 1000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_114925.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_115949.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_121014.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_122039.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_123413.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_124759.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_130333.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/resnet-18/offline/20260113_132243.json --output-dir output/imagenet-c/resnet-18/continuous/threshold-$threshold --response-budget 12.3 --decay-threshold $threshold; sleep 1
# done
#
# 3.3 ViT-Base-Patch16-224, ImageNet-C
# ------------------------------------
# for threshold in 200 400 1000 2000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260203_214745.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260203_232415.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260206_145544.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260206_152314.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260206_173823.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260206_202810.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260206_232747.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260207_002114.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260325_003439.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260325_015504.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-c/vit-base-patch16-224/offline/20260325_030808.json --output-dir output/imagenet-c/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 1
# done
#
# 3.4 ResNet-50, ImageNet-R
# -------------------------
# for threshold in 50 100 200 400 1000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_111944.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_112118.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_112252.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_112426.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_112627.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_112828.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_113040.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_113328.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_113557.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/resnet-50/offline/20260327_113812.json --output-dir output/imagenet-r/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
# done
#
# 3.5 ResNet-50, ImageNet-V2
# --------------------------
# for threshold in 50 100 200 400 1000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_113846.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_113922.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_113958.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114034.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114120.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114205.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114255.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114357.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114452.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/resnet-50/offline/20260327_114544.json --output-dir output/imagenet-v2/resnet-50/continuous/threshold-$threshold --response-budget 39.9 --decay-threshold $threshold; sleep 2
# done
#
# 3.6 ViT-Base-Patch16-224, ImageNet-R
# ------------------------------------
# for threshold in 200 400 1000 2000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_114744.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_114946.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_115147.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_115451.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_115755.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_120111.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_120606.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_121201.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_121618.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_122004.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-r/vit-base-patch16-224/offline/20260327_122356.json --output-dir output/imagenet-r/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
# done
#
# 3.7 ViT-Base-Patch16-224, ImageNet-V2
# -------------------------------------
# for threshold in 200 400 1000 2000; do
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122440.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122526.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122612.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122719.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122827.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_122938.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_123123.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_123328.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_123457.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_123621.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
#     uv run python -m tempora.scripts.evaluate.continuous output/imagenet-v2/vit-base-patch16-224/offline/20260327_123745.json --output-dir output/imagenet-v2/vit-base-patch16-224/continuous/threshold-$threshold --response-budget 105.3 --decay-threshold $threshold; sleep 2
# done


# 4. Amortised evaluation
#    4.1 ResNet-50, ImageNet-C at various overhead budgets
#    4.2 ResNet-18, ImageNet-C at various overhead budgets
#    4.3 ViT-Base-Patch16-224, ImageNet-C at various overhead budgets
#    4.4 ResNet-18, CIFAR-10-C at various overhead budgets (scaled from Raspberry Pi)
#    4.5 ResNet-50, ImageNet-R at various overhead budgets
#    4.6 ResNet-50, ImageNet-V2 at various overhead budgets
#    4.7 ViT-Base-Patch16-224, ImageNet-R at various overhead budgets
#    4.8 ViT-Base-Patch16-224, ImageNet-V2 at various overhead budgets
#
# 4.1 ResNet-50, ImageNet-C at various overhead budgets
# Loop structure: method (outer), budget (inner, ascending)
# Early termination occurs if the remaining overhead > 0 for all distributions; this skips larger budgets.
# -----------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar deyo cmf; do
#     for budget in 1000 2000 4000 8000 16000 32000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised              \
#                 --dataset-name imagenet                                           \
#                 --dataset-dist noise blur weather digital                         \
#                 --output-dir output/imagenet-c/resnet-50/amortised/budget-$budget \
#                 --response-budget 39.9                                            \
#                 --overhead-budget $budget                                         \
#                 --model-arch resnet50                                             \
#                 --method $method                                                  \
#                 --note "Amortised, $budget, 39.9, $method, RN-50, IN-C" 2>&1 | tee /dev/stderr)
#
#         # Check if all distributions have remaining overhead > 0 (budget not exhausted)
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.2 ResNet-18, ImageNet-C at various overhead budgets
# -----------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar; do
#     for budget in 125 250 500 1000 2000 4000 8000 16000 32000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised              \
#                 --dataset-name imagenet                                           \
#                 --dataset-dist noise blur weather digital                         \
#                 --output-dir output/imagenet-c/resnet-18/amortised/budget-$budget \
#                 --response-budget 12.3                                            \
#                 --overhead-budget $budget                                         \
#                 --model-arch resnet18                                             \
#                 --method $method                                                  \
#                 --note "Amortised, $budget, 12.3, $method, RN-18, IN-C" 2>&1 | tee /dev/stderr)
#
#         # Check if all distributions have remaining overhead > 0 (budget not exhausted)
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.3 ViT-Base-Patch16-224, ImageNet-C at various overhead budgets
# ----------------------------------------------------------------
# for method in basic lame neo tent eta shot sar spa deyo cmf zerosiam; do
#     for budget in 2500 5000 10000 20000 40000 80000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised                         \
#                 --dataset-name imagenet                                                      \
#                 --dataset-dist noise blur weather digital                                    \
#                 --output-dir output/imagenet-c/vit-base-patch16-224/amortised/budget-$budget \
#                 --response-budget 105.3                                                      \
#                 --overhead-budget $budget                                                    \
#                 --model-arch vit_base_patch16_224                                            \
#                 --method $method                                                             \
#                 --note "Amortised, $budget, 105.3, $method, ViT-Base-Patch16-224, IN-C" 2>&1 | tee /dev/stderr)
#
#         # Check if all distributions have remaining overhead > 0 (budget not exhausted)
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.4 ResNet-18, CIFAR-10-C at various overhead budgets
# Budgets scaled from Raspberry Pi 5: 37, 74, 148, 296 ms ≈ 12k, 24k, 48k, 96k ms * (1.5 / 486.6)
# -----------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar; do
#     for budget in 37 74 148 296; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised              \
#                 --dataset-name cifar-10                                           \
#                 --dataset-dist noise blur weather digital                         \
#                 --output-dir output/cifar-10-c/resnet-18/amortised/budget-$budget \
#                 --response-budget 1.5                                             \
#                 --overhead-budget $budget                                         \
#                 --model-arch resnet18                                             \
#                 --model-ckpt artefacts/weights/RN18-C10.pt                        \
#                 --method $method                                                  \
#                 --note "Amortised, $budget, 1.5, $method, RN-18, C-10" 2>&1 | tee /dev/stderr)
#         sleep 2
#
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.5 ResNet-50, ImageNet-R at various overhead budgets
# -----------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar cmf deyo; do
#     for budget in 1000 2000 4000 8000 16000 32000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised              \
#                 --dataset-name imagenet                                           \
#                 --dataset-dist imagenet-r                                         \
#                 --output-dir output/imagenet-r/resnet-50/amortised/budget-$budget \
#                 --response-budget 39.9                                            \
#                 --overhead-budget $budget                                         \
#                 --model-arch resnet50                                             \
#                 --method $method                                                  \
#                 --note "Amortised, $budget, 39.9, $method, RN-50, IN-R" 2>&1 | tee /dev/stderr)
#
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.6 ResNet-50, ImageNet-V2 at various overhead budgets
# ------------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar cmf deyo; do
#     for budget in 1000 2000 4000 8000 16000 32000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised               \
#                 --dataset-name imagenet                                            \
#                 --dataset-dist imagenet-v2                                         \
#                 --output-dir output/imagenet-v2/resnet-50/amortised/budget-$budget \
#                 --response-budget 39.9                                             \
#                 --overhead-budget $budget                                          \
#                 --model-arch resnet50                                              \
#                 --method $method                                                   \
#                 --note "Amortised, $budget, 39.9, $method, RN-50, IN-V2" 2>&1 | tee /dev/stderr)
#
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.7 ViT-Base-Patch16-224, ImageNet-R at various overhead budgets
# ----------------------------------------------------------------
# for method in basic lame neo tent eta shot sar spa cmf deyo zerosiam; do
#     for budget in 2500 5000 10000 20000 40000 80000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised                         \
#                 --dataset-name imagenet                                                      \
#                 --dataset-dist imagenet-r                                                    \
#                 --output-dir output/imagenet-r/vit-base-patch16-224/amortised/budget-$budget \
#                 --response-budget 105.3                                                      \
#                 --overhead-budget $budget                                                    \
#                 --model-arch vit_base_patch16_224                                            \
#                 --method $method                                                             \
#                 --note "Amortised, $budget, 105.3, $method, ViT-Base-Patch16-224, IN-R" 2>&1 | tee /dev/stderr)
#
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done
#
# 4.8 ViT-Base-Patch16-224, ImageNet-V2 at various overhead budgets
# -----------------------------------------------------------------
# for method in basic lame neo tent eta shot sar spa cmf deyo zerosiam; do
#     for budget in 2500 5000 10000 20000 40000 80000; do
#         echo "Running: method=$method, budget=$budget"
#         output=$(uv run python -m tempora.scripts.evaluate.amortised                          \
#                 --dataset-name imagenet                                                       \
#                 --dataset-dist imagenet-v2                                                    \
#                 --output-dir output/imagenet-v2/vit-base-patch16-224/amortised/budget-$budget \
#                 --response-budget 105.3                                                       \
#                 --overhead-budget $budget                                                     \
#                 --model-arch vit_base_patch16_224                                             \
#                 --method $method                                                              \
#                 --note "Amortised, $budget, 105.3, $method, ViT-Base-Patch16-224, IN-V2" 2>&1 | tee /dev/stderr)
#
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)
#
#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done


# 5. [Rebuttal] Randomness analysis
#    5.1 Copy seed-2025 baselines
#    5.2 ResNet-50, ImageNet-C, offline (gaussian_noise, seeds 2026 and 2027)
#    5.3 ResNet-50, ImageNet-C, discrete (buffered-100, gaussian_noise, seeds 2026 and 2027)
#    5.4 ResNet-50, ImageNet-C, amortised (budget-1000, gaussian_noise, seeds 2026 and 2027)
#    5.5 ViT-Base-Patch16-224, ImageNet-C, offline (gaussian_noise, seeds 2026 and 2027)
#    5.6 ViT-Base-Patch16-224, ImageNet-C, discrete (buffered-100, gaussian_noise, seeds 2026 and 2027)
#    5.7 ViT-Base-Patch16-224, ImageNet-C, amortised (budget-2500, gaussian_noise, seeds 2026 and 2027)
#
# Note: Seed 2025 results are copied from output/imagenet-c/resnet-50 and from output/imagenet-c/vit-base-patch16-224/
#
# 5.1 Copy seed-2025 baselines
# ----------------------------
# mkdir -p output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025
# mkdir -p output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025
# mkdir -p output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025
# mkdir -p output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025
# mkdir -p output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025
# mkdir -p output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025
#
# cp output/imagenet-c/resnet-50/offline/20260111_003617.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # Basic, RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_005231.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # AdaBN, RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_010828.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # LAME,  RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_012404.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # NEO,   RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_015129.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # Tent,  RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_021856.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # ETA,   RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_025057.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # SHOT,  RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260111_033723.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # SAR,   RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260326_201551.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # CMF,   RN-50, offline
# cp output/imagenet-c/resnet-50/offline/20260326_204931.json output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2025/  # DeYO,  RN-50, offline
#
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_054959.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # Basic, RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_060539.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # AdaBN, RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_062119.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # LAME,  RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_063654.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # NEO,   RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_065301.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # Tent,  RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_070852.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # ETA,   RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_072454.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # SHOT,  RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260111_074101.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # SAR,   RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260326_232044.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # CMF,   RN-50, discrete buffered-100
# cp output/imagenet-c/resnet-50/discrete/buffered-100/20260326_233654.json output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2025/  # DeYO,  RN-50, discrete buffered-100
#
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_121110.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # Basic, RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_122729.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # AdaBN, RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_124334.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # LAME,  RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_125938.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # NEO,   RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_131551.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # Tent,  RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_133154.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # ETA,   RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_134801.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # SHOT,  RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_140420.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # SAR,   RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260327_033328.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # CMF,   RN-50, amortised budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260327_052451.json output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2025/  # DeYO,  RN-50, amortised budget-1000
#
# cp output/imagenet-c/vit-base-patch16-224/offline/20260203_232415.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # Basic,    ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260206_145544.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # LAME,     ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260206_152314.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # NEO,      ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260206_232747.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # Tent,     ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260207_002114.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # ETA,      ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260206_173823.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # SHOT,     ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260206_202810.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # SAR,      ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260203_214745.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # SPA,      ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260325_015504.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # CMF,      ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260325_003439.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # DeYO,     ViT-Base-Patch16-224, offline
# cp output/imagenet-c/vit-base-patch16-224/offline/20260325_030808.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2025/  # ZeroSIAM, ViT-Base-Patch16-224, offline
#
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_204116.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # Basic,    ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_210854.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # LAME,     ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_213624.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # NEO,      ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_220505.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # Tent,     ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_223352.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # ETA,      ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_230238.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # SHOT,     ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260207_233204.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # SAR,      ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260208_000142.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # SPA,      ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260325_052314.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # CMF,      ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260325_045353.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # DeYO,     ViT-Base-Patch16-224, discrete buffered-100
# cp output/imagenet-c/vit-base-patch16-224/discrete/buffered-100/20260325_055229.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2025/  # ZeroSIAM, ViT-Base-Patch16-224, discrete buffered-100
#
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_012908.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # Basic,    ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_015652.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # LAME,     ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_022424.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # NEO,      ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_025229.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # Tent,     ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_062033.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # ETA,      ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_094824.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # SHOT,     ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_131607.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # SAR,      ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_164510.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # SPA,      ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260325_211436.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # CMF,      ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260325_174705.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # DeYO,     ViT-Base-Patch16-224, amortised budget-1000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260326_004233.json output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-1000/seed-2025/  # ZeroSIAM, ViT-Base-Patch16-224, amortised budget-1000
#
# 5.2 ResNet-50, ImageNet-C, offline (gaussian_noise, seeds 2026 and 2027)
# ------------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method basic --note "Randomness, Offline, seed=2026, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method adabn --note "Randomness, Offline, seed=2026, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method lame  --note "Randomness, Offline, seed=2026, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method neo   --note "Randomness, Offline, seed=2026, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method tent  --note "Randomness, Offline, seed=2026, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method eta   --note "Randomness, Offline, seed=2026, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method shot  --note "Randomness, Offline, seed=2026, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method sar   --note "Randomness, Offline, seed=2026, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method cmf   --note "Randomness, Offline, seed=2026, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2026 --seed 2026 --model-arch resnet50 --method deyo  --note "Randomness, Offline, seed=2026, DeYO,  RN-50, IN-C"; sleep 2
#
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method basic --note "Randomness, Offline, seed=2027, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method adabn --note "Randomness, Offline, seed=2027, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method lame  --note "Randomness, Offline, seed=2027, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method neo   --note "Randomness, Offline, seed=2027, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method tent  --note "Randomness, Offline, seed=2027, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method eta   --note "Randomness, Offline, seed=2027, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method shot  --note "Randomness, Offline, seed=2027, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method sar   --note "Randomness, Offline, seed=2027, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method cmf   --note "Randomness, Offline, seed=2027, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/offline/seed-2027 --seed 2027 --model-arch resnet50 --method deyo  --note "Randomness, Offline, seed=2027, DeYO,  RN-50, IN-C"; sleep 2
#
# 5.3 ResNet-50, ImageNet-C, discrete (buffered-100, gaussian_noise, seeds 2026 and 2027)
# ---------------------------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method basic --note "Randomness, Discrete, buffered-100, seed=2026, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method adabn --note "Randomness, Discrete, buffered-100, seed=2026, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method lame  --note "Randomness, Discrete, buffered-100, seed=2026, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method neo   --note "Randomness, Discrete, buffered-100, seed=2026, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method tent  --note "Randomness, Discrete, buffered-100, seed=2026, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method eta   --note "Randomness, Discrete, buffered-100, seed=2026, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method shot  --note "Randomness, Discrete, buffered-100, seed=2026, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method sar   --note "Randomness, Discrete, buffered-100, seed=2026, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method cmf   --note "Randomness, Discrete, buffered-100, seed=2026, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 39.9 --model-arch resnet50 --method deyo  --note "Randomness, Discrete, buffered-100, seed=2026, DeYO,  RN-50, IN-C"; sleep 2
#
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method basic --note "Randomness, Discrete, buffered-100, seed=2027, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method adabn --note "Randomness, Discrete, buffered-100, seed=2027, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method lame  --note "Randomness, Discrete, buffered-100, seed=2027, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method neo   --note "Randomness, Discrete, buffered-100, seed=2027, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method tent  --note "Randomness, Discrete, buffered-100, seed=2027, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method eta   --note "Randomness, Discrete, buffered-100, seed=2027, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method shot  --note "Randomness, Discrete, buffered-100, seed=2027, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method sar   --note "Randomness, Discrete, buffered-100, seed=2027, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method cmf   --note "Randomness, Discrete, buffered-100, seed=2027, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 39.9 --model-arch resnet50 --method deyo  --note "Randomness, Discrete, buffered-100, seed=2027, DeYO,  RN-50, IN-C"; sleep 2
#
# 5.4 ResNet-50, ImageNet-C, amortised (budget-1000, gaussian_noise, seeds 2026 and 2027)
# ---------------------------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method basic --note "Randomness, Amortised, budget-1000, seed=2026, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method adabn --note "Randomness, Amortised, budget-1000, seed=2026, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method lame  --note "Randomness, Amortised, budget-1000, seed=2026, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method neo   --note "Randomness, Amortised, budget-1000, seed=2026, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method tent  --note "Randomness, Amortised, budget-1000, seed=2026, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method eta   --note "Randomness, Amortised, budget-1000, seed=2026, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method shot  --note "Randomness, Amortised, budget-1000, seed=2026, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method sar   --note "Randomness, Amortised, budget-1000, seed=2026, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method cmf   --note "Randomness, Amortised, budget-1000, seed=2026, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2026 --seed 2026 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method deyo  --note "Randomness, Amortised, budget-1000, seed=2026, DeYO,  RN-50, IN-C"; sleep 2
#
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method basic --note "Randomness, Amortised, budget-1000, seed=2027, Basic, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method adabn --note "Randomness, Amortised, budget-1000, seed=2027, AdaBN, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method lame  --note "Randomness, Amortised, budget-1000, seed=2027, LAME,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method neo   --note "Randomness, Amortised, budget-1000, seed=2027, NEO,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method tent  --note "Randomness, Amortised, budget-1000, seed=2027, Tent,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method eta   --note "Randomness, Amortised, budget-1000, seed=2027, ETA,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method shot  --note "Randomness, Amortised, budget-1000, seed=2027, SHOT,  RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method sar   --note "Randomness, Amortised, budget-1000, seed=2027, SAR,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method cmf   --note "Randomness, Amortised, budget-1000, seed=2027, CMF,   RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/resnet-50/amortised/budget-1000/seed-2027 --seed 2027 --response-budget 39.9 --overhead-budget 1000 --model-arch resnet50 --method deyo  --note "Randomness, Amortised, budget-1000, seed=2027, DeYO,  RN-50, IN-C"; sleep 2
#
# 5.5 ViT-Base-Patch16-224, ImageNet-C, offline (gaussian_noise, seeds 2026 and 2027)
# -----------------------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method basic    --note "Randomness, Offline, seed=2026, Basic,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method lame     --note "Randomness, Offline, seed=2026, LAME,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method neo      --note "Randomness, Offline, seed=2026, NEO,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method tent     --note "Randomness, Offline, seed=2026, Tent,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method eta      --note "Randomness, Offline, seed=2026, ETA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method shot     --note "Randomness, Offline, seed=2026, SHOT-IM, ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method sar      --note "Randomness, Offline, seed=2026, SAR,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method spa      --note "Randomness, Offline, seed=2026, SPA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method deyo     --note "Randomness, Offline, seed=2026, DeYO,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method cmf      --note "Randomness, Offline, seed=2026, CMF,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2026 --seed 2026 --model-arch vit_base_patch16_224 --method zerosiam --note "Randomness, Offline, seed=2026, ZeroSIAM, ViT-Base-Patch16-224, IN-C"; sleep 2
#
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method basic    --note "Randomness, Offline, seed=2027, Basic,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method lame     --note "Randomness, Offline, seed=2027, LAME,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method neo      --note "Randomness, Offline, seed=2027, NEO,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method tent     --note "Randomness, Offline, seed=2027, Tent,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method eta      --note "Randomness, Offline, seed=2027, ETA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method shot     --note "Randomness, Offline, seed=2027, SHOT-IM, ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method sar      --note "Randomness, Offline, seed=2027, SAR,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method spa      --note "Randomness, Offline, seed=2027, SPA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method deyo     --note "Randomness, Offline, seed=2027, DeYO,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method cmf      --note "Randomness, Offline, seed=2027, CMF,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/offline/seed-2027 --seed 2027 --model-arch vit_base_patch16_224 --method zerosiam --note "Randomness, Offline, seed=2027, ZeroSIAM, ViT-Base-Patch16-224, IN-C"
#
# 5.6 ViT-Base-Patch16-224, ImageNet-C, discrete (buffered-100, gaussian_noise, seeds 2026 and 2027)
# --------------------------------------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method basic    --note "Randomness, Discrete, buffered-100, seed=2026, Basic,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method lame     --note "Randomness, Discrete, buffered-100, seed=2026, LAME,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method neo      --note "Randomness, Discrete, buffered-100, seed=2026, NEO,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method tent     --note "Randomness, Discrete, buffered-100, seed=2026, Tent,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method eta      --note "Randomness, Discrete, buffered-100, seed=2026, ETA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method shot     --note "Randomness, Discrete, buffered-100, seed=2026, SHOT-IM, ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method sar      --note "Randomness, Discrete, buffered-100, seed=2026, SAR,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method spa      --note "Randomness, Discrete, buffered-100, seed=2026, SPA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method deyo     --note "Randomness, Discrete, buffered-100, seed=2026, DeYO,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method cmf      --note "Randomness, Discrete, buffered-100, seed=2026, CMF,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2026 --seed 2026 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method zerosiam --note "Randomness, Discrete, buffered-100, seed=2026, ZeroSIAM, ViT-Base-Patch16-224, IN-C"; sleep 2
#
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method basic    --note "Randomness, Discrete, buffered-100, seed=2027, Basic,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method lame     --note "Randomness, Discrete, buffered-100, seed=2027, LAME,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method neo      --note "Randomness, Discrete, buffered-100, seed=2027, NEO,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method tent     --note "Randomness, Discrete, buffered-100, seed=2027, Tent,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method eta      --note "Randomness, Discrete, buffered-100, seed=2027, ETA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method shot     --note "Randomness, Discrete, buffered-100, seed=2027, SHOT-IM, ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method sar      --note "Randomness, Discrete, buffered-100, seed=2027, SAR,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method spa      --note "Randomness, Discrete, buffered-100, seed=2027, SPA,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method deyo     --note "Randomness, Discrete, buffered-100, seed=2027, DeYO,    ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method cmf      --note "Randomness, Discrete, buffered-100, seed=2027, CMF,     ViT-Base-Patch16-224, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist gaussian_noise --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/discrete/buffered-100/seed-2027 --seed 2027 --queue-size 1 --interval 105.3 --model-arch vit_base_patch16_224 --method zerosiam --note "Randomness, Discrete, buffered-100, seed=2027, ZeroSIAM, ViT-Base-Patch16-224, IN-C"; sleep 2
#
# 5.7 ViT-Base-Patch16-224, ImageNet-C, amortised (budget-2500, gaussian_noise, seeds 2026 and 2027)
# --------------------------------------------------------------------------------------------------
# for method in basic lame neo tent eta shot sar spa deyo cmf zerosiam; do
#     for seed in 2026 2027; do
#         uv run python -m tempora.scripts.evaluate.amortised                                                          \
#             --dataset-name imagenet                                                                                  \
#             --dataset-dist gaussian_noise                                                                            \
#             --output-dir output/rebuttal/randomness/imagenet-c/vit-base-patch16-224/amortised/budget-2500/seed-$seed \
#             --seed $seed                                                                                             \
#             --response-budget 105.3                                                                                  \
#             --overhead-budget 2500                                                                                   \
#             --model-arch vit_base_patch16_224                                                                        \
#             --method $method                                                                                         \
#             --note "Randomness, Amortised, budget-2500, seed=$seed, $method, ViT-Base-Patch16-224, IN-C"
#         sleep 2
#     done
# done


# 6. [Rebuttal] Amortised ablation, SHOT-IM (norm only) and ETA (no BN reset)
#    6.1 Copy baselines
#    6.2 ResNet-50, ImageNet-C, budget-1000 (strictest)
#    6.3 ResNet-50, ImageNet-C, budget-2000 (second-strictest)
#    6.4 ViT-Base-Patch16-224, ImageNet-C, budget-2500 (strictest)
#    6.5 ViT-Base-Patch16-224, ImageNet-C, budget-5000 (second-strictest)
#
# Note: Baseline SHOT-IM and ETA resutls are copied in from the main imagenet-c runs.
#       ETANoReset is RN-50 only; ViT has no BatchNorm, so the BN-reset ablation is a no-op there.
#
# 6.1 Copy baselines
# ------------------
# mkdir -p output/rebuttal/ablation/resnet-50/amortised/budget-1000
# mkdir -p output/rebuttal/ablation/resnet-50/amortised/budget-2000
# mkdir -p output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-2500
# mkdir -p output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-5000
#
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_133154.json output/rebuttal/ablation/resnet-50/amortised/budget-1000/  # ETA,   RN-50, budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-1000/20260111_134801.json output/rebuttal/ablation/resnet-50/amortised/budget-1000/  # SHOT,  RN-50, budget-1000
# cp output/imagenet-c/resnet-50/amortised/budget-2000/20260111_154109.json output/rebuttal/ablation/resnet-50/amortised/budget-2000/  # ETA,   RN-50, budget-2000
# cp output/imagenet-c/resnet-50/amortised/budget-2000/20260111_155738.json output/rebuttal/ablation/resnet-50/amortised/budget-2000/  # SHOT,  RN-50, budget-2000
#
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_062033.json output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-2500/  # ETA,  ViT, budget-2500
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-2500/20260209_094824.json output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-2500/  # SHOT, ViT, budget-2500
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-5000/20260209_064939.json output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-5000/  # ETA,  ViT, budget-5000
# cp output/imagenet-c/vit-base-patch16-224/amortised/budget-5000/20260209_101729.json output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-5000/  # SHOT, ViT, budget-5000
#
# 6.2 ResNet-50, ImageNet-C, budget-1000 (strictest)
# --------------------------------------------------
# uv run python -m tempora.scripts.evaluate.amortised                       \
#     --dataset-name imagenet                                               \
#     --dataset-dist noise blur weather digital                             \
#     --output-dir output/rebuttal/ablation/resnet-50/amortised/budget-1000 \
#     --response-budget 39.9                                                \
#     --overhead-budget 1000                                                \
#     --model-arch resnet50                                                 \
#     --method shot_norm                                                    \
#     --note "Ablation, Amortised, 1000, 39.9, SHOTNorm, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised                       \
#     --dataset-name imagenet                                               \
#     --dataset-dist noise blur weather digital                             \
#     --output-dir output/rebuttal/ablation/resnet-50/amortised/budget-1000 \
#     --response-budget 39.9                                                \
#     --overhead-budget 1000                                                \
#     --model-arch resnet50                                                 \
#     --method eta_no_reset                                                 \
#     --note "Ablation, Amortised, 1000, 39.9, ETANoReset, RN-50, IN-C"; sleep 2
#
# 6.3 ResNet-50, ImageNet-C, budget-2000 (second-strictest)
# ---------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.amortised                       \
#     --dataset-name imagenet                                               \
#     --dataset-dist noise blur weather digital                             \
#     --output-dir output/rebuttal/ablation/resnet-50/amortised/budget-2000 \
#     --response-budget 39.9                                                \
#     --overhead-budget 2000                                                \
#     --model-arch resnet50                                                 \
#     --method shot_norm                                                    \
#     --note "Ablation, Amortised, 2000, 39.9, SHOTNorm, RN-50, IN-C"; sleep 2
# uv run python -m tempora.scripts.evaluate.amortised                       \
#     --dataset-name imagenet                                               \
#     --dataset-dist noise blur weather digital                             \
#     --output-dir output/rebuttal/ablation/resnet-50/amortised/budget-2000 \
#     --response-budget 39.9                                                \
#     --overhead-budget 2000                                                \
#     --model-arch resnet50                                                 \
#     --method eta_no_reset                                                 \
#     --note "Ablation, Amortised, 2000, 39.9, ETANoReset, RN-50, IN-C"; sleep 2
#
# 6.4 ViT-Base-Patch16-224, ImageNet-C, budget-2500 (strictest)
# -------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.amortised                                  \
#     --dataset-name imagenet                                                          \
#     --dataset-dist noise blur weather digital                                        \
#     --output-dir output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-2500 \
#     --response-budget 105.3                                                          \
#     --overhead-budget 2500                                                           \
#     --model-arch vit_base_patch16_224                                                \
#     --method shot_norm                                                               \
#     --note "Ablation, Amortised, 2500, 105.3, SHOTNorm, ViT-Base-Patch16-224, IN-C"; sleep 2
#
# 6.5 ViT-Base-Patch16-224, ImageNet-C, budget-5000 (second-strictest)
# --------------------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.amortised                                  \
#     --dataset-name imagenet                                                          \
#     --dataset-dist noise blur weather digital                                        \
#     --output-dir output/rebuttal/ablation/vit-base-patch16-224/amortised/budget-5000 \
#     --response-budget 105.3                                                          \
#     --overhead-budget 5000                                                           \
#     --model-arch vit_base_patch16_224                                                \
#     --method shot_norm                                                               \
#     --note "Ablation, Amortised, 5000, 105.3, SHOTNorm, ViT-Base-Patch16-224, IN-C"; sleep 2