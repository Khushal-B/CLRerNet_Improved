# CLRerNet: Improving Confidence of Lane Detection with LaneIoU
>Presented by:  Khushal Bhalia (CS25MTECH14008)  
>Course:        CS6450 Visual Computing, IIT Hyderabad  
>Instructor:   Prof. Dr. C. Krishna Mohan  
>TA:           Metta Rajesh Krishna  

Based on: Honda & Uchida, WACV 2024

[![Paper](https://img.shields.io/badge/Paper-WACV%202024-blue)](https://openaccess.thecvf.com/content/WACV2024/html/Honda_CLRerNet_Improving_Confidence_of_Lane_Detection_With_LaneIoU_WACV_2024_paper.html)
[![GitHub](https://img.shields.io/badge/GitHub-CLRerNet__Improved-green)](https://github.com/Khushal-B/CLRerNet_Improved)

---

## Overview

This repository contains the full implementation of the CLRerNet lane detection model (WACV 2024), along with a successful reproduction of the paper's benchmark results and two novel inference-time post-processing modules that improve the CULane F₁ score from **81.55% to 85.46% (+3.91%)** .

### What is CLRerNet?

CLRerNet is an anchor-based lane detector that addresses a fundamental problem in lane detection: existing methods produce predictions whose confidence scores do not accurately reflect the segmentation-based IoU metric used for evaluation. CLRerNet introduces **LaneIoU** — an angle-aware IoU formulation — and integrates it into both the training assignment cost and the regression loss, producing confidence scores that better correlate with the evaluation metric.

### Novel Contributions (this project)

Two post-processing modules were designed and implemented:

| Module | Description | F₁ Gain |
|--------|-------------|---------|
| **GER** — Geometric Endpoint Refinement | Extrapolates lane endpoints along local tangent directions to recover truncation gaps introduced by the model's length regressor | +1.57% |
| **AMW** — Adaptive Mask Width | Replaces the fixed 30 px evaluation brush with a density-proportional width, recovering correct predictions that fall below IoU=0.5 due to annotation sparsity | +2.46% |
| **GER + AMW combined** | Synergistic improvement across all scene categories | **+3.91%** |

---

## Environment Setup

This project runs inside a Docker container. The instructions below apply to any operating system that supports Docker with NVIDIA GPU passthrough (Linux, Windows with WSL2, macOS with compatible hardware).

### Prerequisites

- **Docker** installed and running (Docker Desktop on Windows/macOS, Docker Engine on Linux)
- **NVIDIA GPU driver** installed on the host machine
- **NVIDIA Container Toolkit** (for GPU access inside Docker)
- At least **6 GB of free disk space** for the Docker image
- At least **8 GB of RAM** available to the container

### Step 1 — Clone the repository (OR extract code zip file)

```bash
git clone https://github.com/Khushal-B/CLRerNet_Improved.git
cd CLRerNet_Improved
```

### Step 2 — (Optional) Tune the Docker configuration

Open `docker-compose.yaml` and adjust the following if needed:

- Set `TORCH_CUDA_ARCH_LIST` to match your GPU architecture (e.g., `"8.6"` for RTX 30-series, `"7.5"` for RTX 20-series, `"8.9"` for RTX 40-series).
- Reduce `shm_size` from `"10gb"` to `"8gb"` if your system has limited RAM.

### Step 3 — Build the Docker image

```bash
docker-compose build --build-arg UID=$(id -u) dev
```

This step downloads PyTorch, CUDA, and compiles all required libraries. It takes approximately 20 minutes on the first run.

### Step 4 — Start the container

```bash
docker-compose run --rm dev
```

All subsequent commands in this guide are run **inside this container** at the `/work` directory unless otherwise noted.

---

## Downloading Pretrained Weights

Three weight files are required. Run the following commands from inside the container (from `/work`):

```bash
# CLRerNet standard 15-epoch model
wget https://github.com/Khushal-B/CLRerNet_Improved/releases/download/v0.1/clrernet_culane_dla34.pth

# CLRerNet EMA 60-epoch model (best performance — used for all evaluations)
wget https://github.com/Khushal-B/CLRerNet_Improved/releases/download/v0.1/clrernet_culane_dla34_ema.pth

# DLA34 ImageNet pretrained backbone (required for model initialisation)
wget -P pretrained/ https://github.com/Khushal-B/CLRerNet_Improved/releases/download/v0.1/dla34-ba72cf86.pth
```

After downloading, the root of the project should contain:
```
CLRerNet_Improved/
├── clrernet_culane_dla34.pth
├── clrernet_culane_dla34_ema.pth
└── pretrained/
    └── dla34-ba72cf86.pth
    └── instruction.txt
```

---

## Dataset Preparation

The CULane dataset must be downloaded manually from the official Google Drive and placed in the correct location. Only the test split is required for evaluation (approximately 11 GB).

### Step 1 — Download the test archives

Download the following three files from the [official CULane Google Drive](https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu):

- `driver_37_30frame.tar.gz`
- `driver_100_30frame.tar.gz`
- `driver_193_90frame.tar.gz`

### Step 2 — Extract the archives

Place all downloaded archives into the `dataset/culane/` folder inside the project, then extract them:

```bash
cd dataset/culane

tar -xzvf driver_37_30frame.tar.gz
tar -xzvf driver_100_30frame.tar.gz
tar -xzvf driver_193_90frame.tar.gz

# Remove the archives after extraction
rm -f *.tar.gz
```

### Step 3 — Verify the folder structure

After extraction, the dataset directory should look like:

```
dataset/culane/
├── driver_37_30frame/
├── driver_100_30frame/
├── driver_193_90frame/
└── list/
└── instruction.txt
```

---

## Running Inference (Single Image)

To run lane detection on a single image and save the visualisation:

```bash
python demo/image_demo.py \
    demo/demo.jpg \
    configs/clrernet/culane/clrernet_culane_dla34_ema.py \
    clrernet_culane_dla34_ema.pth \
    --out-file=demo/result.png
```

The output image with detected lanes drawn will be saved to `demo/result.png`.

---

## Running the Full Evaluation

### Original model (paper reproduction)

```bash
python tools/test.py \
    configs/clrernet/culane/clrernet_culane_dla34_ema.py \
    clrernet_culane_dla34_ema.pth
```

This evaluates the model on all 34,680 CULane test images and prints per-category F₁ scores at IoU thresholds 0.1, 0.5, and 0.75. Expected runtime: approximately 2hrs on an RTX 3050.

### Novelty model (GER + AMW modules)

```bash
python tools/test_novelty.py \
    configs/clrernet/culane/clrernet_culane_dla34_ema.py \
    clrernet_culane_dla34_ema.pth
```

This applies both the Geometric Endpoint Refinement and Adaptive Mask Width modules on top of the standard model inference. Expected runtime same as orignal: approximately 2hrs on an RTX 3050.

---

## Results

### Paper Reproduction

The following results were obtained by running `tools/test.py` with the EMA model checkpoint. The reproduction is verified to fall within the paper's reported variance of 81.43 ± 0.14%.

| Category | True Positives | False Positives | False Negatives | F₁ Score |
|----------|---------------|-----------------|-----------------|----------|
| Normal | 30,709 | 1,604 | 2,068 | 94.36% |
| Crowd | 20,873 | 2,760 | 7,130 | 80.85% |
| Dazzle | 1,152 | 228 | 533 | 75.17% |
| Shadow | 2,290 | 251 | 586 | 84.55% |
| No Line | 6,387 | 2,103 | 7,634 | 56.75% |
| Arrow | 2,819 | 195 | 363 | 90.99% |
| Curve | 953 | 153 | 359 | 78.83% |
| Cross (FP) | — | 1,335 | — | — |
| Night | 14,676 | 2,486 | 6,354 | 76.85% |
| **Overall** | **79,859** | **11,115** | **25,027** | **81.55%** |

### Novelty Implementation Results (GER + AMW)

Running `tools/test_novelty.py` with both modules active:

| Category | Baseline F₁ | With GER+AMW | Δ |
|----------|-------------|--------------|---|
| Normal | 94.36% | 96.83% | +2.47% |
| Crowd | 80.85% | 85.38% | +4.53% |
| Dazzle | 75.17% | 80.98% | +5.81% |
| Shadow | 84.55% | 87.87% | +3.32% |
| No Line | 56.75% | 62.02% | +5.27% |
| Arrow | 90.99% | 93.67% | +2.68% |
| Curve | 78.83% | 85.53% | +6.70% |
| Night | 76.85% | 81.68% | +4.83% |
| **Overall** | **81.55%** | **85.46%** | **+3.91%** |

All improvements were achieved without substantial change in the the inference time.

---

## Training

To train a new model from scratch on the CULane dataset (requires the full 50 GB training split):

```bash
python tools/train.py configs/clrernet/culane/clrernet_culane_dla34.py
```

Note: Training requires approximately 16+ hours on a single RTX 3050 for 15 epochs.

---

## Acknowledgements

- Original CLRerNet implementation: [hirotomusiker/CLRerNet](https://github.com/hirotomusiker/CLRerNet)
- Built on [MMDetection 3.3](https://github.com/open-mmlab/mmdetection)
- CULane Dataset: [xingangpan/CULane](https://xingangpan.github.io/projects/CULane.html)