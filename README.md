# LeverageDet (DINOv3) — 非官方复现

> 本仓库是论文（arXiv: https://arxiv.org/pdf/2509.12995）的 DINOv3 版本非官方复现。复现采用冻结 DINOv3 视觉骨干 + 线性分类头（Linear Probe）的方式完成 AIGC 图像检测。复现实验结果与论文报告存在差异，详见下文“复现实验与差异”。

## 概述
- 核心做法：冻结 `DINOv3` 骨干，仅训练一个二分类的线性头来区分 `real` 与 `fake`。
- 数据格式：`torchvision.datasets.ImageFolder`，目录名包含 `real/fake` 或 `0_*/1_*` 均可自动映射。
- 实现栈：Python + PyTorch + HuggingFace Transformers。

## 快速开始
### 环境依赖（示例）
```
conda create -n leveragedet python=3.10 -y
conda activate leveragedet
# 按你的 CUDA/OS 安装匹配版本的 PyTorch/torchvision（参见 https://pytorch.org）
# 例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers tqdm pillow
```

### 预训练权重（DINOv3）
- 将 `--model_id` 指向可用的 DINOv3 权重：
  - 本地目录（例如 `dinov3-vitl16-pretrain-lvd1689m/`），或
  - 在线模型名称（需联网从 Hub 加载）。
- 本仓库默认忽略大体积权重与检查点（见 `.gitignore`），因此 `dinov3-vitl16-pretrain-lvd1689m/` 不会被提交到 GitHub。

## 数据准备
- 使用 `ImageFolder` 目录结构，二分类目录名包含 `real/fake` 或 `0_*/1_*`。示例：
```
<train_root>/
  real/ ...
  fake/ ...

<val_root>/
  0_real/ ...
  1_fake/ ...
```
- 如果目录名不满足上述规则，脚本会报错提示；或使用脚本中提供的鲁棒映射逻辑进行自动推断。

## 训练
- 单卡线性探测（冻结骨干，仅训练线性头）：
```
python train_linear_fixedval.py \
  --train_root /path/to/your/train \
  --val_root   /path/to/your/val   \
  --model_id   dinov3-vitl16-pretrain-lvd1689m \
  --epochs 10 --batch_size 16 --lr 1e-3 --amp
```
- 最优验证结果对应的线性头会保存到 `checkpoints/dinov3_linear_head_best.pt`。
- 若提供 `--resume`，脚本会先进行一次验证作为基线；当为默认 10 轮且存在 checkpoint 时，会自动扩展到 90 轮继续训练。
- 也可使用封装脚本：
```
bash train.sh
```
> 其中多卡入口名为 `train_linear_fixedval_multi.py`（如未包含可自行扩展或只使用单卡脚本）。

## 推理 / 评测
- 对一个或多个数据根目录下的子集进行评测，并汇总 Acc / Precision / Recall / F1：
```
python infer_linear_fixedval.py \
  --data_roots /path/A /path/B \
  --model_id dinov3-vitl16-pretrain-lvd1689m \
  --ckpt checkpoints/dinov3_linear_head_best.pt \
  --batch_size 64 --amp
```
- 当 `--ckpt` 缺失时会尝试回退到 `checkpoints/dinov3_linear_head_best_multi.pt`；均缺失则以随机初始化头评测（性能显著下降）。

## 复现实验与差异
- 本仓库为论文（https://arxiv.org/pdf/2509.12995）“非官方复现”，实验设置与实现细节可能与原论文存在差异，因而复现结果与论文报告不同。
- 可能差异来源（举例）：
  - 预处理/增广：本实现默认仅做 resize + normalize，无复杂数据增强；
  - 训练超参：优化器、学习率、批大小、轮数、随机种子等；
  - 骨干/特征：DINOv3 具体变体、`pooler_output` vs `CLS` token 的取用；
  - 数据切分：训练/验证集来源、分布与比例差异。

## 实验结果（占位，待补充）
- Chameleon（Val/Test）：Acc = [TBD]，Precision = [TBD]，Recall = [TBD]，F1 = [TBD]
- 其它数据集：
  - [Dataset-1]: Acc = [TBD] | P = [TBD] | R = [TBD] | F1 = [TBD]
  - [Dataset-2]: Acc = [TBD] | P = [TBD] | R = [TBD] | F1 = [TBD]

## 目录结构（简要）
- `train_linear_fixedval.py`：单卡训练（冻结骨干 + 线性头）。
- `infer_linear_fixedval.py`：多数据集评测与汇总。
- `train.sh`：封装脚本（根据可见 GPU 选择入口）。
- `checkpoints/`：训练产物（默认忽略入库）。
- `dinov3-vitl16-pretrain-lvd1689m/`：本地 DINOv3 权重目录（不会提交到 GitHub）。

## 致谢
- 论文原文（arXiv）：https://arxiv.org/pdf/2509.12995
- DINOv3、Transformers 与 PyTorch 生态。
- 参考项目：SAFE（https://github.com/graydove/SAFE）。

