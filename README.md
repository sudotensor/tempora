# Tempora

This is the official project repository for [Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](https://arxiv.org/abs/2602.06136) by [Sudarshan Sreeram](https://www.linkedin.com/in/sudarshan-sreeram/), [Young D. Kwon](https://theyoungkwon.github.io/research.html), and [Cecilia Mascolo](https://www.cl.cam.ac.uk/~cm542/) (ICML 2026). 

Tempora is a framework for evaluating test-time adaptation (TTA) under temporal constraints. It comprises temporal scenarios, evaluation protocols, and time-contingent utility metrics. We instantiate this framework with three metrics for distinct scenarios: 

1. **Discrete utility**: Asynchronous streams with hard deadlines; batches arriving while the pipeline is occupied are skipped.
2. **Continuous utility**: User-led pacing with hyperbolic decay; late predictions lose value proportional to response delay.
3. **Amortised utility**: Budget-constrained overhead; adaptation proceeds until the budget is exhausted, then the model freezes.

By applying Tempora to 11 TTA methods, we find that rank instability persists across 750+ temporal evaluations spanning diverse datasets, models, and hardware platforms; i.e., conventional rankings do not predict rankings under temporal pressure. The figure below visualises this persistence. Cells show the highest-utility method among nine TTA methods tested on ImageNet-C with ResNet-50 (BN). _Offline_: No time constraints. _Temporal_: ❶ Discrete (hard deadlines), ❷ Continuous (latency penalties), and ❸ Amortised (budgeted overhead). Rows reveal rank instability: the best method changes
under temporal pressure. Similarly, columns show no consistent winner within any scenario. Aggregate benchmarks obscure these corruption-level dynamics. 

<div align="center" style="background-color: white; padding: 10px; display: inline-block;">
  <img src="https://github.com/user-attachments/assets/6e734bb2-11f4-4997-a661-71805621ecdd" width="445" alt="tempora-summary">
</div>

To reproduce the results in our paper[^1] or apply/extend Tempora to your settings, please follow the guidelines below. All our evaluation logfiles are available for comparison or analysis on [Google Drive](https://drive.google.com/drive/folders/1El1yaZ02Ms7sInZtWuFlbtHKZzLqjq3v?usp=sharing) (2.1 GB).

## Installation & Usage

Please follow the listed instructions to download this repository and start using Tempora. A Linux machine is recommended, along with Python 3.12+ and the [uv](https://docs.astral.sh/uv/) package manager. 

```
git clone https://github.com/sudotensor/tempora.git
cd ./tempora
uv sync
```

### Repository Structure

Our repository follows the structure signposted below. 

```
tempora/
├── tempora/
│   ├── common/
│   │   ├── methods/       # TTA method implementations
│   │   ├── models/        # Model architectures
│   │   ├── datasets/      # Dataset loaders
│   │   ├── recorders/     # Utility recorders
│   │   ├── constants.py
│   │   └── utils.py       # Contains method setup
│   └── scripts/
│       └── evaluate/      # Evaluation scripts
│       └── run.sh         # Evaluation runner
└── pyproject.toml         # Dependencies
```

**Dataset Setup.** Tempora is configured with support for CIFAR-10/100(-C) and ImageNet(-C/R/A/V2); adaptation methods and model architectures are only selectively compatible with these datasets. Please download the datasets from their respective sources and place them in a `datasets` folder at the project root (symlink recommended); we expect ImageNet-V2 folders to be named by integer class index (0-999). ImageNet validation data (optional) requires preprocessing with `valprep.sh` ([Link](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)) to organise images into class-wise folders. We expect the filetree to look like this once configured:

```
datasets/
├── imagenet-c/
├── imagenet/
│   └── val/
├── cifar-10-c/
└── cifar-10/
    └── cifar-10-batches-py/  # From https://www.cs.toronto.edu/~kriz/cifar.html
```

**Model Setup.** ResNet-18/50 (BN) are automatically downloaded from Torchvision's pre-trained model hub; i.e., ImageNet1-V1 weights: `resnet50-0676ba61.pth` ([Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50)) and `resnet18-f37072fd.pth` ([Link](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)). ViT-B/16 and ResNet-50 (GN)[^2] are also automatically downloaded, this time from Timm (via HuggingFace); i.e., `vit_base_patch16_224.augreg2_in21k_ft_in1k` ([Link](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)) and `resnet50_gn.a1h_in1k` ([Link](https://huggingface.co/timm/resnet50_gn.a1h_in1k)). If evaluating on CIFAR-10(-C), please download the pre-trained checkpoints for ResNet-18 from [huyvnphan/PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master). Models are patched to optionally return features before the classifier layer.

**Method Setup.** We implement 11 adaptation methods. Implementing a new one is straightforward: (1) create a new file in the methods folder, (2) update the export definitions, (3) create a new case block in the method setup (`tempora/common/utils.py`), and (4) add the method name to the `METHODS` constant (`tempora/common/constants.py`).

### Running Evaluations

Below, we provide example commands to run each of the four kinds of evaluation scripts. Each evaluation produces a JSON file containing (1) `arguments`, evaluation configuration, and (2) `results`, per-batch metrics (predictions, timing, accuracy). Please refer to `tempora/scripts/run.sh` for commands we used conduct our evaluation.

```bash
# Offline evaluation
uv run python -m tempora.scripts.evaluate.offline \
    --model-arch resnet-50                        \
    --method eta                                  \
    --dataset-name imagenet                       \
    --dataset-root <path/to/datasets>             \
    --dataset-dist noise blur weather digital     \
    --output-dir output/offline

# Discrete utility (utilisation ρ controlled via interval)
uv run python -m tempora.scripts.evaluate.discrete \
    --model-arch resnet-50                         \
    --method eta                                   \
    --interval 39.9                                \
    --queue-size 1                                 \
    --dataset-name imagenet                        \
    --dataset-root <path/to/datasets>              \
    --dataset-dist noise blur weather digital      \
    --output-dir output/discrete

# Continuous utility (budget T in ms)
uv run python -m tempora.scripts.evaluate.continuous \
   --response-budget 39.9                            \
   --decay-threshold 100                             \
   --output-dir output/continuous                    \
   <path/to/offline/result>

# Amortised utility (budget B in ms)
uv run python -m tempora.scripts.evaluate.amortised \
    --model-arch resnet-50                          \
    --method eta                                    \
    --overhead-budget 1000                          \
    --dataset-name imagenet                         \
    --dataset-root <path/to/datasets>               \
    --dataset-dist noise blur weather digital       \
    --output-dir output/amortised
```

## Correspondence

Please contact Sudarshan Sreeram at sudotensor [at] gmail [dot] com (or) raise an issue from the "Issues" tab.

## Citation

If you use Tempora or find it relevant to your research, please consider citing:

```
@inproceedings{sreeram2026tempora,
  title     = {Tempora: Characterising the Time-Contingent Utility of Online Test-Time Adaptation},
  author    = {Sreeram, Sudarshan and Kwon, Young D. and Mascolo, Cecilia},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
} 
```

## Acknowledgements

Our code relies on the (revised) implementations of several adaptation methods. We provide links to the source code and paper in each method definition file (`tempora/common/methods`).


## License

MIT License. See LICENSE for details.

[^1]: Please note that our results were obtained on an Nvidia RTX 4080 Founders Edition GPU and a Raspberry Pi 5 (16 GB). 
[^2]: We do not use this model in our evaluation. Similarly, we also include support for MobileNet-V2 and MobileNet-V3-Small, both downloaded through Torchvision's model hub, but do not use it in our evaluation. Please let us know if you face any issue using these models.
