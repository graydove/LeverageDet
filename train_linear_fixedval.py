import argparse, os, random, time, json
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm isn't installed
    def tqdm(x, *args, **kwargs):
        return x

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

@dataclass
class Meter:
    correct:int=0; total:int=0
    tp:int=0; tn:int=0; fp:int=0; fn:int=0
    def update(self, logits: torch.Tensor, y: torch.Tensor):
        pred = logits.argmax(dim=1)
        self.correct += (pred==y).sum().item()
        self.total   += y.numel()
        self.tp += ((pred==1)&(y==1)).sum().item()
        self.tn += ((pred==0)&(y==0)).sum().item()
        self.fp += ((pred==1)&(y==0)).sum().item()
        self.fn += ((pred==0)&(y==1)).sum().item()
    @property
    def acc(self): return self.correct / max(1,self.total)
    @property
    def precision(self): 
        return self.tp / max(1, (self.tp+self.fp))
    @property
    def recall(self):
        return self.tp / max(1, (self.tp+self.fn))
    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2*p*r/max(1e-12,(p+r))

# 统一标签：0=real, 1=fake；兼容 {real,fake} 与 {0_real,1_fake}
def standardize_binary_labels(ds: datasets.ImageFolder):
    new_samples, new_targets = [], []
    for path, _idx in ds.samples:
        cname = os.path.basename(os.path.dirname(path)).lower()
        if ("fake" in cname) or cname.startswith("1_") or cname == "1":
            y = 1
        elif ("real" in cname) or cname.startswith("0_") or cname == "0":
            y = 0
        else:
            raise ValueError(
                f"无法从文件夹名推断标签：'{cname}'。请使用包含'real/fake'或'0_/1_'的目录名。")
        new_samples.append((path, y))
        new_targets.append(y)
    ds.samples = new_samples
    ds.targets = new_targets
    ds.classes = ["real", "fake"]
    ds.class_to_idx = {"real":0, "fake":1}
    return ds

def make_collate(processor, device):
    def _collate(batch):
        imgs, labels = list(zip(*batch))
        inputs = processor(images=list(imgs), return_tensors="pt")
        pixel_values = inputs["pixel_values"]  # keep on CPU; move in training loop
        labels = torch.tensor(labels, dtype=torch.long)  # CPU; move in training loop
        return pixel_values, labels
    return _collate

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=bias)
    def forward(self, x):
        return self.fc(x)

# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(backbone, head, loader, use_amp: bool):
    backbone.eval(); head.eval()
    meter = Meter()
    has_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if has_cuda else torch.float32
    amp_device_type = "cuda" if has_cuda else "cpu"
    for pixel_values, y in loader:
        pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
        y = y.to(next(head.parameters()).device, non_blocking=True)
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            outputs = backbone(pixel_values=pixel_values)
            feats = (outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None
                     else outputs.last_hidden_state[:, 0, :])
            logits = head(feats)
        meter.update(logits, y)
    return meter

def train_one_epoch(backbone, head, loader, optimizer, use_amp: bool, epoch: int | None = None, total_epochs: int | None = None):
    backbone.eval()  # 冻结
    head.train()
    ce = nn.CrossEntropyLoss()
    total_loss, meter = 0.0, Meter()
    has_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and has_cuda))
    amp_dtype = torch.bfloat16 if has_cuda else torch.float32
    amp_device_type = "cuda" if has_cuda else "cpu"

    progress_desc = f"Epoch {epoch}/{total_epochs} [train]" if (epoch is not None and total_epochs is not None) else "Train"
    loop = tqdm(loader, total=len(loader), desc=progress_desc, leave=False)
    for pixel_values, y in loop:
        pixel_values = pixel_values.to(next(backbone.parameters()).device, non_blocking=True)
        y = y.to(next(head.parameters()).device, non_blocking=True)
        with torch.no_grad():
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
                outputs = backbone(pixel_values=pixel_values)
                feats = (outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None
                         else outputs.last_hidden_state[:, 0, :])
        # 线性头前向 + 反传
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=(use_amp and has_cuda)):
            logits = head(feats)
            loss = ce(logits, y)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and has_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item() * y.size(0)
        meter.update(logits, y)
        # 更新进度条信息
        avg_loss = total_loss / max(1, meter.total)
        if hasattr(loop, "set_postfix"):
            loop.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{meter.acc*100:.2f}%")

    return total_loss / max(1, meter.total), meter

def infer_feat_dim(backbone, processor, device):
    # 用零张量走一次前向获取维度（仅用于确定线性头大小）
    size = getattr(processor, "size", None)
    H = W = 224
    if isinstance(size, dict):
        if "shortest_edge" in size:
            H = W = int(size["shortest_edge"]) or 224
        elif "height" in size and "width" in size:
            H = int(size["height"]) or 224
            W = int(size["width"]) or 224
        elif "shortest_side" in size:
            H = W = int(size["shortest_side"]) or 224
    elif isinstance(size, int):
        H = W = int(size)
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        H, W = int(size[0]), int(size[1])
    dummy = torch.zeros(1, 3, H, W, device=device)
    with torch.no_grad():
        out = backbone(pixel_values=dummy)
        feats = (out.pooler_output if getattr(out, "pooler_output", None) is not None
                 else out.last_hidden_state[:,0,:])
    return feats.shape[-1]

# ---------------- Processor Fallback ----------------
try:
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception:
    TF = None
    class InterpolationMode:
        BILINEAR = None

class SimpleImageProcessor:
    def __init__(self, size=224, image_mean=(0.485,0.456,0.406), image_std=(0.229,0.224,0.225), rescale_factor=1.0):
        # size can be int or dict with height/width
        if isinstance(size, dict):
            self.size = {"height": int(size.get("height", 224)), "width": int(size.get("width", 224))}
        elif isinstance(size, int):
            self.size = {"height": int(size), "width": int(size)}
        elif isinstance(size, (tuple, list)) and len(size)==2:
            self.size = {"height": int(size[0]), "width": int(size[1])}
        else:
            self.size = {"height": 224, "width": 224}
        self.image_mean = torch.tensor(image_mean).view(3,1,1)
        self.image_std  = torch.tensor(image_std).view(3,1,1)
        self.rescale_factor = float(rescale_factor)

    @staticmethod
    def from_pretrained(model_id_or_path: str):
        # Try to read preprocessor_config.json then config.json
        size = 224; mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225); rescale=1/255
        pp_path = os.path.join(model_id_or_path, 'preprocessor_config.json')
        if os.path.isfile(pp_path):
            try:
                with open(pp_path, 'r') as f:
                    cfg = json.load(f)
                if isinstance(cfg.get('size'), dict):
                    sz = cfg['size']
                    size = {"height": int(sz.get('height', 224)), "width": int(sz.get('width', 224))}
                elif isinstance(cfg.get('size'), int):
                    size = int(cfg['size'])
                mean = tuple(cfg.get('image_mean', mean))
                std  = tuple(cfg.get('image_std', std))
                rescale = float(cfg.get('rescale_factor', rescale))
            except Exception:
                pass
        else:
            cfg_path = os.path.join(model_id_or_path, 'config.json')
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                    if 'image_size' in cfg:
                        size = int(cfg['image_size'])
                except Exception:
                    pass
        return SimpleImageProcessor(size=size, image_mean=mean, image_std=std, rescale_factor=rescale)

    def __call__(self, images, return_tensors="pt"):
        tensors = []
        H, W = int(self.size['height']), int(self.size['width'])
        for img in images:
            # Convert to tensor with expected normalization
            if TF is None:
                # Minimal fallback using PIL -> tensor-like via torch
                import PIL.Image
                if isinstance(img, PIL.Image.Image):
                    img = img.convert('RGB')
                    # naive resize via torchvision missing; try numpy
                    import numpy as np
                    arr = np.array(img.resize((W, H)))
                    t = torch.from_numpy(arr).permute(2,0,1).float() * (self.rescale_factor if self.rescale_factor!=0 else 1.0)
                    t = (t - self.image_mean*255*self.rescale_factor) / (self.image_std*255*self.rescale_factor)
                else:
                    t = img
            else:
                import PIL.Image
                if isinstance(img, PIL.Image.Image):
                    img = img.convert('RGB')
                else:
                    # assume tensor [C,H,W] in 0..1
                    pass
                # Resize
                if isinstance(img, torch.Tensor):
                    # convert to PIL to use high-quality resize if needed
                    img = TF.to_pil_image(img)
                img = TF.resize(img, [H, W], interpolation=InterpolationMode.BILINEAR)
                t = TF.to_tensor(img)  # [0,1]
                if abs(self.rescale_factor - 1/255) > 1e-8:
                    t = t / (1/255) * self.rescale_factor
                # Normalize
                t = (t - self.image_mean) / self.image_std
            tensors.append(t)
        batch = torch.stack(tensors, dim=0)
        return {"pixel_values": batch}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, default="./datasets/GenImage_train/imagenet_ai_0419_sdv4/train/", required=True, help="训练集根目录（ImageFolder：包含 real/fake 或 0_real/1_fake）")
    parser.add_argument("--val_root", type=str, default="./datasets/Chameleon/test/", required=True, help="验证集根目录（你的 Chameleon 路径：/home/.../Chameleon/test/）")
    parser.add_argument("--model_id", type=str, default="dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="启用AMP(bfloat16/float16)推理与训练以省显存")
    parser.add_argument("--resume", type=str, default="", help="从checkpoint继续训练的路径，例如 checkpoints/dinov3_linear_head_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 自动断点续训：若未指定 --resume 且默认checkpoint存在，则启用；并在仍为默认10轮时改为90轮
    if not args.resume:
        default_ckpt = os.path.join("checkpoints", "dinov3_linear_head_best.pt")
        if os.path.isfile(default_ckpt):
            args.resume = default_ckpt
            print(f"[Auto-Resume] Using checkpoint: {args.resume}")
    if args.resume and args.epochs == 10:
        args.epochs = 90
        print("[Auto-Resume] epochs set to 90 for continued training")

    # 1) 模型与处理器
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"[Warn] AutoImageProcessor not available for '{args.model_id}' ({e}). Using SimpleImageProcessor fallback.")
        processor = SimpleImageProcessor.from_pretrained(args.model_id)
    backbone  = AutoModel.from_pretrained(args.model_id)
    backbone.to(device)
    for p in backbone.parameters(): p.requires_grad = False

    # 2) 数据（无数据增强）
    train_set = datasets.ImageFolder(args.train_root)
    val_set   = datasets.ImageFolder(args.val_root)
    train_set = standardize_binary_labels(train_set)
    val_set   = standardize_binary_labels(val_set)

    collate = make_collate(processor, device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # 3) 线性头
    feat_dim = infer_feat_dim(backbone, processor, device)
    head = LinearHead(in_dim=feat_dim, num_classes=2).to(device)
    # 3.1) 若提供 --resume，则载入线性头权重
    if args.resume:
        if os.path.isfile(args.resume):
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt.get("head", ckpt)
            try:
                head.load_state_dict(state, strict=True)
                print(f"[Resume] Loaded head weights from {args.resume}")
            except Exception as e:
                print(f"[Resume] Strict load failed: {e}; trying non-strict...")
                missing, unexpected = head.load_state_dict(state, strict=False)
                print(f"[Resume] Loaded with missing={list(missing)} unexpected={list(unexpected)}")
        else:
            print(f"[Resume] Checkpoint not found: {args.resume}")

    # 4) 优化器
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 5) 训练 + 固定验证（Chameleon）
    # 若从checkpoint恢复，先做一次基线验证，确保best_state至少等于当前权重
    best_acc, best_state = 0.0, None
    if args.resume:
        base_meter = evaluate(backbone, head, val_loader, use_amp=args.amp)
        best_acc = base_meter.acc
        best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        print(f"[Resume] Baseline Val Acc: {best_acc*100:.2f}% | P {base_meter.precision*100:.2f}% | R {base_meter.recall*100:.2f}% | F1 {base_meter.f1*100:.2f}%")
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_meter = train_one_epoch(backbone, head, train_loader, optimizer, use_amp=args.amp, epoch=epoch, total_epochs=args.epochs)
        val_meter = evaluate(backbone, head, val_loader, use_amp=args.amp)
        dt = time.time() - t0

        if val_meter.acc > best_acc:
            best_acc = val_meter.acc
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

        print(f"[Epoch {epoch:02d}] {dt:.1f}s | "
              f"train_acc {tr_meter.acc*100:.2f} | "
              f"val_acc {val_meter.acc*100:.2f} | "
              f"val_P {val_meter.precision*100:.2f} R {val_meter.recall*100:.2f} F1 {val_meter.f1*100:.2f}")

    # 6) 保存最优头并输出最终混淆矩阵
    os.makedirs("checkpoints", exist_ok=True)
    if best_state is not None:
        torch.save({"head": best_state, "model_id": args.model_id, "feat_dim": feat_dim},
                   "checkpoints/dinov3_linear_head_best.pt")
        print(f"Saved: checkpoints/dinov3_linear_head_best.pt (best val_acc={best_acc*100:.2f})")
        head.load_state_dict(best_state)

    final = evaluate(backbone, head, val_loader, use_amp=args.amp)
    print("\nConfusion Matrix (rows=GT, cols=Pred):")
    print(f"            Pred: real    Pred: fake")
    print(f"GT: real     {final.tn:>6d}      {final.fp:>6d}")
    print(f"GT: fake     {final.fn:>6d}      {final.tp:>6d}")
    print(f"\nFinal Val Acc: {final.acc*100:.2f}% | "
          f"P {final.precision*100:.2f}% | R {final.recall*100:.2f}% | F1 {final.f1*100:.2f}%")

if __name__ == "__main__":
    main()
