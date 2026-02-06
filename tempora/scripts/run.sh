#!/usr/bin/env bash
# Experiment runner file for "Tempora: Characterising the time-contingent utility of online test-time adaptation" (ICML 2025 submission)


# Instructions:
# 1. Please uncomment to run the evaluations.
# 2. Run this file from the project root directory. 
#    Alternatively, prefix your PYTHONPATH with the project root and modify the evaluation blocks with the filepath:
#    PYTHONPATH=<path to root>:PYTHONPATH uv run python <path to file>.py [options]
# 3. Adjust the standard inference latency, thresholds, and budgets according to measurements on your hardware platform
#    Use the offline script for "basic" first and compute the mean and std. dev. of latencies across all distributions
#    Guideline: standard inference latency = mean + 6 * std. dev. 


# 1. Offline evaluation
#    1.1 ResNet-50, ImageNet-C
#    1.2 ResNet-18, ImageNet-C
#    1.3 ViT-Base-Patch16-224, ImageNet-C
#
# 1.1 ResNet-50, ImageNet-C
# -------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method basic --note "Offline, Basic, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method adabn --note "Offline, AdaBN, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method lame  --note "Offline, LAME,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method neo   --note "Offline, NEO,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method tent  --note "Offline, Tent,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method eta   --note "Offline, ETA,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method shot  --note "Offline, SHOT,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/offline --model-arch resnet50 --method sar   --note "Offline, SAR,   RN-50, IN-C"
#
# 1.2 ResNet-18, ImageNet-C
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
# ------------------------------------
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method basic --note "Offline, Basic, ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method lame  --note "Offline, LAME,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method neo   --note "Offline, NEO,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method tent  --note "Offline, Tent,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method eta   --note "Offline, ETA,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method shot  --note "Offline, SHOT,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method sar   --note "Offline, SAR,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.offline --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/offline --model-arch vit_base_patch16_224 --method spa   --note "Offline, SPA,   ViT-Base-Patch16-224, IN-C"


# 2. Discrete evaluation
#    2.1 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-C
#    2.2 Buffered, ResNet-50, ImageNet-C at various utilisation levels
#    2.3 Unbuffered (Alfarra et al.), ResNet-18, ImageNet-C
#    2.4 Buffered, ResNet-18, ImageNet-C at 100% utilisation only
#    2.5 Unbuffered (Alfarra et al.), ViT-B-Patch16-224, ImageNet-C
#    2.6 Buffered, ViT-Base-Patch16-224, ImageNet-C at various utilisation levels
#
# 2.1 Unbuffered (Alfarra et al.), ResNet-50, ImageNet-C
# ------------------------------------------------------
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method basic --note "Discrete, No queue, 39.9, Basic, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method adabn --note "Discrete, No queue, 39.9, AdaBN, RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method lame  --note "Discrete, No queue, 39.9, LAME,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method neo   --note "Discrete, No queue, 39.9, NEO,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method tent  --note "Discrete, No queue, 39.9, Tent,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method eta   --note "Discrete, No queue, 39.9, ETA,   RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method shot  --note "Discrete, No queue, 39.9, SHOT,  RN-50, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/resnet-50/discrete/unbuffered --queue-size 0 --interval 39.9 --method sar   --note "Discrete, No queue, 39.9, SAR,   RN-50, IN-C"
#
# 2.2 Buffered, ResNet-50, ImageNet-C at various utilisation levels
# iat = {1, sqrt(2), 2, 2 * sqrt(2), 4} * 39.9 ms; this corresponds to rho in {100%, 70%, 50%, 35%, 25%} utilisation
# -----------------------------------------------------------------
# for info in "39.9:100" "56.4:70" "79.8:50" "112.8:35" "159.6:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic adabn lame neo tent eta shot sar; do
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
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method basic --note "Discrete, No queue, 105.3, Basic, ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method lame  --note "Discrete, No queue, 105.3, LAME,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method neo   --note "Discrete, No queue, 105.3, NEO,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method tent  --note "Discrete, No queue, 105.3, Tent,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method eta   --note "Discrete, No queue, 105.3, ETA,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method shot  --note "Discrete, No queue, 105.3, SHOT,  ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method sar   --note "Discrete, No queue, 105.3, SAR,   ViT-Base-Patch16-224, IN-C"
# uv run python -m tempora.scripts.evaluate.discrete --dataset-name imagenet --dataset-dist noise blur weather digital --output-dir output/imagenet-c/vit-base-patch16-224/discrete/unbuffered --queue-size 0 --interval 105.3 --model-arch vit_base_patch16_224 --method spa   --note "Discrete, No queue, 105.3, SPA,   ViT-Base-Patch16-224, IN-C"
#
# 2.6 Buffered, ViT-Base-Patch16-224, ImageNet-C at various utilisation levels
# iat = {1, sqrt(2), 2, 2 * sqrt(2), 4} * 105.3 ms; this corresponds to rho in {100%, 70%, 50%, 35%, 25%} utilisation
# -----------------------------------------------------------------
# for info in "105.3:100" "148.9:70" "210.6:50" "297.8:35" "421.2:25"; do
#     iat=$(echo $info | cut -d: -f1)
#     rho=$(echo $info | cut -d: -f2)
#     for method in basic lame neo tent eta shot sar spa; do
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


# 3. Continuous evaluation
#    3.1 ResNet-50, ImageNet-C 
#    3.2 ResNet-18, ImageNet-C
#    3.3 ViT-Base-Patch16-224, ImageNet-C
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
# done


# 4. Amortised utilisation
#    4.1 ResNet-50, ImageNet-C at various overhead budgets
#    4.2 ResNet-18, ImageNet-C at various overhead budgets
#    4.3 ViT-Base-Patch16-224, ImageNet-C at various overhead budgets
#
# 4.1 ResNet-50, ImageNet-C at various overhead budgets
# Loop structure: method (outer), budget (inner, ascending)
# Early termination occurs if the remaining overhead > 0 for all distributions; this skips larger budgets.
# -----------------------------------------------------
# for method in basic adabn lame neo tent eta shot sar; do
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
# for method in basic lame neo tent eta shot sar spa; do
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
       
#         # Check if all distributions have remaining overhead > 0 (budget not exhausted)
#         lines=$(echo "$output" | grep -E "^Remaining\s+:" | sed 's/,//g')
#         total=$(echo "$lines" | wc -l)
#         n_rem=$(echo "$lines" | awk -F: '{gsub(/[^0-9.]/, "", $2); if ($2 + 0 > 0) print}' | wc -l)

#         echo "Distributions: $total, With remaining overhead: $n_rem"
#         if [ "$n_rem" -eq "$total" ] && [ "$total" -gt 0 ]; then
#             echo "All distributions have remaining overhead. Skipping larger budgets for $method."
#             break
#         fi
#     done
# done