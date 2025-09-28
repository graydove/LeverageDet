# LeverageDet (DINOv3) — Unofficial Reproduction
> This repository contains an unofficial DINOv3-based reproduction of the paper (arXiv: https://arxiv.org/pdf/2509.12995). We freeze a DINOv3 vision backbone and train a lightweight linear classification head (linear probe) for AIGC image detection. Reproduction results differ from those reported in the paper; see “Reproduction Notes & Differences”.

## Overview
- Approach: freeze the `DINOv3` backbone and train a binary linear head to classify `real` vs `fake`.
- Data format: `torchvision.datasets.ImageFolder`. Folder names containing `real/fake` or `0_*/1_*` are automatically mapped to labels.
- Stack: Python + PyTorch + HuggingFace Transformers.

## Quick Start
### Environment (example)
```shell
conda create -n leveragedet python=3.10 -y
conda activate leveragedet
# Install the PyTorch/torchvision wheels that match your CUDA/OS (see https://pytorch.org)
# e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### DINOv3 Pretrained Weights
- Point `--model_id` to a DINOv3 model:
  - a local directory (e.g., `dinov3-vitl16-pretrain-lvd1689m/`), or
  - a model repo name (will download from the Hub if available).
- Large weight folders and checkpoints are ignored by Git (see `.gitignore`), so `dinov3-vitl16-pretrain-lvd1689m/` will NOT be pushed to GitHub.

## Data Preparation
- Use `ImageFolder`-style directories with two classes. Example:
```
<train_root>/
  real/ ...
  fake/ ...

<val_root>/
  0_real/ ...
  1_fake/ ...
```
- If class folder names do not match the above patterns, the scripts attempt robust mapping and otherwise raise a clear error.

## Training
- Single-GPU linear probe (frozen backbone, train only the linear head):
```shell
python train_linear_fixedval.py \
  --train_root /path/to/your/train \
  --val_root   /path/to/your/val   \
  --model_id   dinov3-vitl16-pretrain-lvd1689m \
  --epochs 100 --batch_size 128 --lr 1e-5 --amp
```
- The best head is saved to `checkpoints/dinov3_linear_head_best.pt`.
- If `--resume` is provided, the script runs a baseline validation first. When epochs are the default 10 and a checkpoint exists, it auto-expands to 90 epochs for continued training.
- You can also use the wrapper script:
```shell
bash train.sh
```
> Note: `train.sh` mentions a multi-GPU entry (`train_linear_fixedval_multi.py`). If that file is not present in your tree, use the single-GPU script above or provide your own DDP entry.

## Inference / Evaluation
- Evaluate one or more dataset roots (each root containing multiple subfolders), and print Acc / Precision / Recall / F1:
```shell
python infer_linear_fixedval.py \
  --data_roots /path/A /path/B \
  --model_id dinov3-vitl16-pretrain-lvd1689m \
  --ckpt checkpoints/dinov3_linear_head_best.pt \
  --batch_size 64 --amp
```
- If `--ckpt` is missing, the script tries `checkpoints/dinov3_linear_head_best_multi.pt`. If none are found, it runs with a randomly initialized head (expect poor performance).

## Reproduction Notes & Differences
- This is an unofficial reproduction of the paper (https://arxiv.org/pdf/2509.12995). Implementation details and experimental settings may differ, so results will not match exactly.
- Potential sources of difference include but are not limited to:
  - Preprocessing/Augmentation: the default implementation uses only resize + normalize; no heavy augmentations.
  - Training hyperparameters: optimizer, learning rate, batch size, number of epochs, random seed, etc.
  - Backbone/features: the specific DINOv3 variant and feature selection (`pooler_output` vs CLS token).
  - Data splits: training/validation dataset composition and distributions.

## Results
### Facebook/dinov3-vitl16-pretrain-lvd1689m + MLP
- **Chameleon (Val/Test): Acc  85.26%  |  P 80.28%  |  R 87.02%  |  F1 83.52%**

- Other datasets:
  - **GenImage: Acc 87.32%**
    
    | ADM   | BigGAN | Glide | Midjourney | SD_v_1_4 | SD_v_1_5 | VQDM  | WuKong |
    | ----- | ------ | ----- | ---------- | -------- | -------- | ----- | ------ |
    | 62.82 | 77.20  | 88.98 | 87.51      | 98.70    | 98.36    | 88.22 | 96.73  |
    
  - **AIGI-Holmes P3: Acc 91.54%**
    
    | FLUX  | Infinity | Janus | Janus-Pro-1B | Janus-Pro-7B | LlamaGen | PixArt-XL | SD35-L | Show-o | VAR   |
    | ----- | -------- | ----- | ------------ | ------------ | -------- | --------- | ------ | ------ | ----- |
    | 91.46 | 97.66    | 91.01 | 95.58        | 89.99        | 97.90    | 98.30     | 80.75  | 97.90  | 74.84 |
  
  - **Real-Robust Dataset: Acc 91.54%**
  
    | Original | Transfer | Redigital |
    | -------- | -------- | --------- |
    | 87.31    | 69.85    | 68.06     |
  
    

### Facebook/dinov3-vit7b16-pretrain-lvd1689m + MLP

- **Chameleon (Val/Test): Acc  93.12%  |  P 95.66%  |  R 87.95%  |  F1 91.64%**

- Other datasets:
  
  - **GenImage: Acc 96.47%**
  
  | ADM   | BigGAN | Glide | Midjourney | SD_v_1_4 | SD_v_1_5 | VQDM  | WuKong |
  | ----- | ------ | ----- | ---------- | -------- | -------- | ----- | ------ |
  | 85.80 | 98.40  | 96.97 | 92.53      | 99.83    | 99.64    | 99.29 | 99.33  |
  
  - **AIGI-Holmes P3: Acc 96.58%**
  
  | FLUX  | Infinity | Janus | Janus-Pro-1B | Janus-Pro-7B | LlamaGen | PixArt-XL | SD35-L | Show-o | VAR   |
  | ----- | -------- | ----- | ------------ | ------------ | -------- | --------- | ------ | ------ | ----- |
  | 92.28 | 99.79    | 99.73 | 99.30        | 98.48        | 99.79    | 99.87     | 84.77  | 99.46  | 92.31 |
  
  - **Real-Robust Dataset: Acc 85.33%**
  
  | Original | Transfer | Redigital |
  | -------- | -------- | --------- |
  | 93.89    | 83.70    | 78.41     |

## Layout
- `train_linear_fixedval.py`: single-GPU training (frozen backbone + linear head).
- `infer_linear_fixedval.py`: multi-dataset evaluation and summary reporting.
- `train.sh`: wrapper that chooses single vs multi-GPU entry based on visible devices.
- `checkpoints/`: training artifacts (ignored by Git).
- `dinov3-vitl16-pretrain-lvd1689m/`: local DINOv3 weights folder (ignored by Git and not pushed).

## Acknowledgments
- Paper (arXiv): https://arxiv.org/pdf/2509.12995
- DINOv3, Transformers, and PyTorch ecosystems.
