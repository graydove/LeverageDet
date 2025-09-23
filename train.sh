#!/usr/bin/env sh
set -e

# 固定使用 2,3 两张卡（避免继承外部环境导致跑到其他卡上）
export CUDA_VISIBLE_DEVICES=2

PYTHON=${PYTHON:-python}

# 计算使用进程数：优先依据 CUDA_VISIBLE_DEVICES，其次查询实际可见GPU数
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  # 统计逗号数量+1
  COMMAS=$(printf "%s" "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c | awk '{print $1}')
  NPROC=$((COMMAS + 1))
else
  NPROC=$($PYTHON - <<'PY'
import os
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(1)
PY
  )
fi

echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}; nproc_per_node=${NPROC}"

TRAIN_ROOT=${TRAIN_ROOT:-/home/zhouxinghui/ssd/datasets/GenImage_train/imagenet_ai_0419_sdv4/train/}
VAL_ROOT=${VAL_ROOT:-/home/zhouxinghui/ssd/datasets/Chameleon/test/}
MODEL_ID=${MODEL_ID:-dinov3-vitl16-pretrain-lvd1689m}
EPOCHS=${EPOCHS:-11}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-1e-6}
WD=${WD:-0.0}

if [ "$NPROC" -ge 2 ]; then
  # 多卡并行，使用 torchrun 启动 DDP 脚本
  if command -v torchrun >/dev/null 2>&1; then
    torchrun --nproc_per_node="$NPROC" train_linear_fixedval_multi.py \
      --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT" \
      --model_id "$MODEL_ID" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
      --lr "$LR" --weight_decay "$WD" --amp
  else
    # 兼容旧式入口
    "$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" train_linear_fixedval_multi.py \
      --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT" \
      --model_id "$MODEL_ID" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
      --lr "$LR" --weight_decay "$WD" --amp
  fi
else
  # 单卡运行常规脚本
  "$PYTHON" train_linear_fixedval.py \
    --train_root "$TRAIN_ROOT" --val_root "$VAL_ROOT" \
    --model_id "$MODEL_ID" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --lr "$LR" --weight_decay "$WD" --amp --resume ./checkpoints/dinov3_linear_head_best.pt
fi
