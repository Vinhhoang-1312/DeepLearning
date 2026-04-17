"""
generate_notebooks.py
Chạy script này 1 lần để tạo ra 3 file .ipynb:
  01_Segmentation_Comparison.ipynb
  02_Classification_Comparison.ipynb
  03_EndToEnd_Comparison.ipynb
"""

import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": text}

def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU"
        },
        "cells": cells
    }

def save(nb, name):
    path = os.path.join(BASE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅  Created: {name}")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — SEGMENTATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

SEG_CELLS = [

md("# 🔬 Notebook 1: Segmentation Comparison — CNN vs RNN vs Transformer\n"
   "Train 3 custom segmentation models on 80×80 RBC images + binary masks.\n"
   "**Metrics:** Dice Coefficient, Mean IoU"),

code("""\
# ── 0. Install dependencies (run once on Colab) ──────────────────────────────
# !pip install -q torchmetrics scikit-learn matplotlib seaborn tqdm pillow
"""),

code("""\
# ── 1. Imports & GPU check ───────────────────────────────────────────────────
import os, random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(42)
"""),

code("""\
# ── 2. Config ────────────────────────────────────────────────────────────────
# ▶ LOCAL path (dùng khi connect Colab Runtime đến local kernel)
DATASET_BASE = r"c:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Học sâu\\Project\\Dataset"

# ▶ COLAB path (uncomment nếu upload data lên Google Drive)
# from google.colab import drive; drive.mount('/content/drive')
# DATASET_BASE = "/content/drive/MyDrive/Project/Dataset"

SLIDE_DIRS = [
    "Elsafty_RBCs_for_Segmentation_and_Detection_Slide_2",
    "Elsafty_RBCs_for_Segmentation_and_Detection_Slide_3",
]

IMG_SIZE   = 80
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
MAX_SAMPLES = 5000   # tổng ảnh dùng để train (giảm nếu Colab hết RAM)

RESULTS_DIR = os.path.join(os.path.dirname(DATASET_BASE), "results", "segmentation")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Config OK")
"""),

code("""\
# ── 3. Dataset ────────────────────────────────────────────────────────────────
class RBCSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=80):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return self.img_tf(img), (self.mask_tf(mask) > 0.5).float()


def collect_paths(base, slide_dirs, max_samples):
    import zipfile
    imgs, masks = [], []
    for sd in slide_dirs:
        sd_path = Path(base) / sd
        
        # 1. Tự động kiểm tra file ảnh (nhanh, dùng next() thay vì load nguyên list gây quá tải RAM)
        has_imgs = next(sd_path.rglob("*.png"), None) is not None or next(sd_path.rglob("*.jpg"), None) is not None
        
        if not has_imgs:
            print(f"Tiến hành giải nén tự động cho: {sd} ...")
            zips = list(sd_path.glob("*.zip"))
            if not zips:
                print(f"⚠ Bỏ qua {sd_path}: Thư mục không có ảnh và không có file .zip!")
                continue
            for zip_path in zips:
                print(f"  Đang giải nén: {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(sd_path)
                    
        # 2. Phân loại ảnh/mask nhanh gọn
        print(f"Đang duyệt ảnh trong thư mục: {sd} ...")
        img_files, mask_files = [], []
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
        
        for p in sd_path.rglob("*.*"):
            if not p.is_file() or p.name.startswith("."): 
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
                
            path_str = str(p.parent).lower()
            if "mask" in path_str or "label" in path_str:
                mask_files.append(p)
            elif "crop" in path_str or "image" in path_str:
                img_files.append(p)

        if not img_files or not mask_files:
            print(f"⚠ Bỏ qua {sd}: Không phân loại được ảnh/mask (Tìm thấy: {len(img_files)} crop, {len(mask_files)} mask).")
            continue
            
        # 3. Ghép cặp ảnh-mask
        mask_dict = {m.stem: str(m) for m in mask_files}
        matched = 0
        for img_p in img_files:
            stem = img_p.stem
            if stem in mask_dict:
                imgs.append(str(img_p))
                masks.append(mask_dict[stem])
                matched += 1
                
        if matched == 0:
            print(f"⚠ Cảnh báo {sd}: Tìm thấy {len(img_files)} crop và {len(mask_files)} mask nhưng không có cặp nào khớp tên!")
            
    if len(imgs) == 0:
        raise ValueError("Data rỗng! Không có cặp ảnh/mask nào được phân loại thành công.")
        
    idx = random.sample(range(len(imgs)), min(max_samples, len(imgs)))
    return [imgs[i] for i in idx], [masks[i] for i in idx]


img_paths, mask_paths = collect_paths(DATASET_BASE, SLIDE_DIRS, MAX_SAMPLES)
print(f"Total samples: {len(img_paths)}")

dataset  = RBCSegDataset(img_paths, mask_paths, IMG_SIZE)
n_train  = int(0.7 * len(dataset))
n_val    = int(0.15 * len(dataset))
n_test   = len(dataset) - n_train - n_val
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(DEVICE=="cuda"))
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
print(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
"""),

md("## 🏗 Model Architectures\n### Pipeline A — Custom CNN (Mini U-Net)"),

code("""\
# ── 4A. CNN — Mini U-Net ─────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x)
        max_o, _ = torch.max(x, dim=1, keepdim=True)
        avg_o = torch.mean(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([max_o, avg_o], dim=1))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch)
    def forward(self, x): return self.cbam(self.net(x))

class MiniUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[16, 32, 64]):
        super().__init__()
        self.encoders  = nn.ModuleList()
        self.pools     = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        self.upconvs   = nn.ModuleList()

        prev = in_ch
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f

        self.bottleneck = DoubleConv(prev, prev * 2)
        prev = prev * 2

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev, f, 2, 2))
            self.decoders.append(DoubleConv(f * 2, f))
            prev = f

        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)

model_cnn = MiniUNet().to(DEVICE)
params_cnn = sum(p.numel() for p in model_cnn.parameters())
print(f"CNN U-Net params: {params_cnn:,}")
"""),

md("### Pipeline B — Custom RNN (ConvLSTM U-Net)"),

code("""\
# ── 4B. RNN — ConvLSTM U-Net ─────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        i, f, g, o = self.gates(combined).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, B, H, W, device):
        return (torch.zeros(B, self.hidden_ch, H, W, device=device),
                torch.zeros(B, self.hidden_ch, H, W, device=device))


class ConvLSTMUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[16, 32, 64]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs  = nn.ModuleList()

        prev = in_ch
        for f in features:
            self.encoders.append(DoubleConv(prev, f))
            self.pools.append(nn.MaxPool2d(2))
            prev = f

        bn_ch = prev * 2
        self.bn_conv     = nn.Conv2d(prev, bn_ch, 1)
        self.conv_lstm   = ConvLSTMCell(bn_ch, bn_ch)

        prev = bn_ch
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev, f, 2, 2))
            self.decoders.append(DoubleConv(f * 2, f))
            prev = f

        self.head = nn.Conv2d(prev, out_ch, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x); skips.append(x); x = pool(x)

        x  = self.bn_conv(x)
        B, C, H, W = x.shape
        h, c = self.conv_lstm.init_hidden(B, H, W, x.device)
        # treat spatial rows as time steps
        for t in range(H):
            row = x[:, :, t:t+1, :].expand(-1, -1, H, W)
            h, c = self.conv_lstm(row, h, c)
        x = h

        skips = skips[::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)

model_rnn = ConvLSTMUNet().to(DEVICE)
params_rnn = sum(p.numel() for p in model_rnn.parameters())
print(f"ConvLSTM U-Net params: {params_rnn:,}")
"""),

md("### Pipeline C — Custom Transformer (ViT-UNet)"),

code("""\
# ── 4C. Transformer — ViT-UNet ───────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=80, patch_size=8, in_ch=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.h_patches  = img_size // patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)                          # B,E,H/P,W/P
        B, E, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)     # B, N, E
        return tokens, H, W


class TransBlock(nn.Module):
    def __init__(self, dim=64, heads=4, mlp_ratio=3, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTUNet(nn.Module):
    def __init__(self, img_size=80, patch_size=8, in_ch=3, out_ch=1,
                 embed_dim=64, depth=4, heads=4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        n = self.patch.n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, n, embed_dim) * 0.02)
        self.blocks    = nn.Sequential(*[TransBlock(embed_dim, heads) for _ in range(depth)])
        self.norm      = nn.LayerNorm(embed_dim)
        # CNN decoder to restore spatial resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 2, 2),  # ×2
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 2, 2),         # ×2
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 2, 2),         # ×2 → 80×80 for patch=8
            nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, out_ch, 1),
        )

    def forward(self, x):
        tokens, H, W = self.patch(x)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)                    # B,N,E
        B, N, E = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(B, E, H, W)
        return self.decoder(spatial)

model_vit = ViTUNet().to(DEVICE)
params_vit = sum(p.numel() for p in model_vit.parameters())
print(f"ViT-UNet params: {params_vit:,}")

print(f"\\nParameter Summary:")
print(f"  CNN  U-Net : {params_cnn:>8,}")
print(f"  RNN  U-Net : {params_rnn:>8,}")
print(f"  ViT  U-Net : {params_vit:>8,}")
"""),

md("## 🏋 Training"),

code("""\
# ── 5. Loss & Metrics ────────────────────────────────────────────────────────
def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred)
    num    = 2 * (pred * target).sum(dim=(1,2,3))
    den    = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (1 - num / den).mean()

def focal_loss(pred, target, alpha=0.8, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = alpha * (1 - pt)**gamma * bce
    return loss.mean()

def bce_dice_loss(pred, target):
    # Dùng Focal Loss kết hợp Dice Loss cho hiệu ứng tối ưu viền tế bào
    return focal_loss(pred, target) + dice_loss(pred, target)

def dice_score(pred, target, eps=1e-6):
    pred   = (torch.sigmoid(pred) > 0.5).float()
    num    = 2 * (pred * target).sum()
    den    = pred.sum() + target.sum() + eps
    return (num / den).item()

def iou_score(pred, target, eps=1e-6):
    pred   = (torch.sigmoid(pred) > 0.5).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter + eps
    return (inter / union).item()
"""),

code("""\
# ── 6. Training loop ─────────────────────────────────────────────────────────
def train_model(model, name, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ReduceLR theo mode 'max' vì ta giám sát Dice Score
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    early_stop_patience = 10
    min_delta = 0.003
    epochs_no_improve = 0

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    best_dice, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(imgs), masks)
            loss.backward()
            # Kỹ thuật: Gradient Clipping giúp tránh exploding gradients (đặc biệt cho RNN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # ── validate ──
        model.eval()
        val_dice_sum, val_iou_sum = 0., 0.
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_dice_sum += dice_score(preds, masks)
                val_iou_sum  += iou_score(preds, masks)

        avg_dice = val_dice_sum / len(val_dl)
        avg_iou  = val_iou_sum  / len(val_dl)
        scheduler.step(avg_dice)

        history["train_loss"].append(avg_loss)
        history["val_dice"].append(avg_dice)
        history["val_iou"].append(avg_iou)

        if avg_dice > best_dice + min_delta:
            best_dice  = avg_dice
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if True:  # Log every epoch as requested
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"[{name}] Epoch {epoch:3d}/{epochs} | LR {curr_lr:.1e} | "
                  f"Loss {avg_loss:.4f} | Dice {avg_dice:.4f} | IoU {avg_iou:.4f}")
        
        # Kỹ thuật: Early Stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping tại epoch {epoch} (Không cải thiện sau {early_stop_patience} epochs)")
            break

    # load best
    model.load_state_dict(best_state)
    torch.save(best_state, f"{RESULTS_DIR}/{name}_best.pt")
    print(f"✅ {name} best Dice = {best_dice:.4f}")
    return history
"""),

code("""\
# ── 7. Train all 3 models ────────────────────────────────────────────────────
t0 = time.time()
hist_cnn = train_model(model_cnn, "CNN_UNet")
print(f"⏱ CNN training time: {(time.time()-t0)/60:.1f} min\\n")

t0 = time.time()
hist_rnn = train_model(model_rnn, "RNN_UNet")
print(f"⏱ RNN training time: {(time.time()-t0)/60:.1f} min\\n")

t0 = time.time()
hist_vit = train_model(model_vit, "ViT_UNet")
print(f"⏱ ViT training time: {(time.time()-t0)/60:.1f} min")
"""),

md("## 📊 Evaluation & Comparison"),

code("""\
# ── 8. Test evaluation ───────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    dice_sum, iou_sum = 0., 0.
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            dice_sum += dice_score(preds, masks)
            iou_sum  += iou_score(preds, masks)
    n = len(loader)
    return dice_sum / n, iou_sum / n

results = {}
for m, name in [(model_cnn, "CNN U-Net"), (model_rnn, "RNN U-Net"), (model_vit, "ViT-UNet")]:
    d, i = evaluate(m, test_dl)
    results[name] = {"Dice": d, "IoU": i}
    print(f"{name:15s} → Dice: {d:.4f}  IoU: {i:.4f}")
"""),

code("""\
# ── 9. Plot training curves & comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for hist, label, color in [
    (hist_cnn, "CNN U-Net",  "steelblue"),
    (hist_rnn, "RNN U-Net",  "coral"),
    (hist_vit, "ViT-UNet",   "seagreen"),
]:
    axes[0].plot(hist["train_loss"], label=label, color=color)
    axes[1].plot(hist["val_dice"],   label=label, color=color)
    axes[2].plot(hist["val_iou"],    label=label, color=color)

for ax, title in zip(axes, ["Train Loss", "Val Dice", "Val IoU"]):
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("Segmentation: CNN vs RNN vs Transformer", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=150)
plt.show()

# Bar chart comparison
names  = list(results.keys())
dices  = [results[n]["Dice"] for n in names]
ious   = [results[n]["IoU"]  for n in names]
x      = np.arange(len(names))
width  = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, dices, width, label="Dice", color=["steelblue","coral","seagreen"])
ax.bar(x + width/2, ious,  width, label="IoU",  color=["steelblue","coral","seagreen"], alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Test Set: Dice & IoU Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/metric_comparison.png", dpi=150)
plt.show()
"""),

code("""\
# ── 10. Visual prediction samples ────────────────────────────────────────────
def show_predictions(models_names, loader, n=3):
    imgs_b, masks_b = next(iter(loader))
    imgs_b, masks_b = imgs_b[:n].to(DEVICE), masks_b[:n].to(DEVICE)

    fig, axes = plt.subplots(n, 2 + len(models_names), figsize=(14, n * 3.5))
    for i in range(n):
        img_np  = imgs_b[i].cpu().permute(1,2,0).numpy()
        img_np  = (img_np * 0.5 + 0.5).clip(0, 1)
        mask_np = masks_b[i].cpu().squeeze().numpy()

        axes[i][0].imshow(img_np);   axes[i][0].set_title("Input")
        axes[i][1].imshow(mask_np, cmap="gray"); axes[i][1].set_title("GT Mask")

        for j, (model, name) in enumerate(models_names):
            with torch.no_grad():
                pred = torch.sigmoid(model(imgs_b[i:i+1])).squeeze().cpu().numpy()
            axes[i][2+j].imshow(pred > 0.5, cmap="gray")
            axes[i][2+j].set_title(name)

        for ax in axes[i]: ax.axis("off")

    plt.suptitle("Segmentation Predictions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/prediction_samples.png", dpi=150)
    plt.show()

show_predictions(
    [(model_cnn, "CNN"), (model_rnn, "RNN"), (model_vit, "ViT")],
    test_dl
)
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — CLASSIFICATION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

CLS_CELLS = [

md("# 🏷 Notebook 2: Classification Comparison — CNN vs RNN vs Transformer\n"
   "Train 3 custom classifiers on 80×80 labeled RBC images (9 classes).\n"
   "**Metrics:** Accuracy, Macro F1, Confusion Matrix"),

code("""\
# !pip install -q scikit-learn matplotlib seaborn tqdm pillow
"""),

code("""\
import os, random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
torch.manual_seed(42)
"""),

code("""\
# ── Config ────────────────────────────────────────────────────────────────────
DATASET_BASE = r"c:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Học sâu\\Project\\Dataset"
# from google.colab import drive; drive.mount('/content/drive')
# DATASET_BASE = "/content/drive/MyDrive/Project/Dataset"

CLS_DIR       = os.path.join(DATASET_BASE, "Elsafty_RBCs_for_Classification", "Cropped images")
IMG_SIZE      = 80
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-3
SAMPLES_PER_CLASS = 1500   # set lower (e.g. 800) if RAM is limited

RESULTS_DIR = os.path.join(os.path.dirname(DATASET_BASE), "results", "classification")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Config OK")
"""),

code("""\
# ── Dataset ───────────────────────────────────────────────────────────────────
import zipfile
cls_dir_path = Path(CLS_DIR)
import zipfile, shutil
cls_dir_path = Path(CLS_DIR)
# 1. Tự động giải nén tất cả các file zip nếu có
for z in cls_dir_path.glob("*.zip"):
    print(f"Bung file nén: {z.name} ...")
    with zipfile.ZipFile(z, 'r') as zf:
        zf.extractall(cls_dir_path)

# 2. Dọn dẹp: Xóa các thư mục rỗng hoặc không chứa ảnh (thường do zip tạo ra thư mục thừa)
for d in list(cls_dir_path.iterdir()):
    if d.is_dir() and not d.name.startswith('.'):
        has_png = any(d.rglob("*.png"))
        if not has_png:
            print(f"🗑️ Đang xóa thư mục rỗng/thừa: {d.name}")
            shutil.rmtree(d)

# 3. CHỈ lấy những thư mục thực sự chứa ảnh .png và chuẩn hóa danh sách class
CLASSES = sorted([d.name for d in cls_dir_path.iterdir() if d.is_dir() and any(d.glob("*.png"))])
print(f"Classes ({len(CLASSES)}): {CLASSES}")
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

class RBCClsDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        base = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)]
        aug  = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15)]
        self.tf = transforms.Compose((aug if augment else []) + base)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.tf(Image.open(path).convert("RGB")), label


def load_samples(cls_dir, spc, classes):
    samples = []
    for cls in classes:
        folder = Path(cls_dir) / cls
        files  = sorted(folder.glob("*.png"))
        if len(files) == 0:
             print(f"⚠ Lỗi: Thư mục {cls} không có ảnh .png nào!")
             continue
        chosen = random.sample(list(files), min(spc, len(files)))
        for f in chosen:
            samples.append((str(f), CLASS2IDX[cls]))
    random.shuffle(samples)
    if len(samples) == 0:
        raise ValueError("Data rỗng! Không collect được ảnh nào cho Classification.")
    return samples

all_samples = load_samples(CLS_DIR, SAMPLES_PER_CLASS, CLASSES)
print(f"Total samples: {len(all_samples)}")

n_train = int(0.7 * len(all_samples))
n_val   = int(0.15 * len(all_samples))
n_test  = len(all_samples) - n_train - n_val

train_s, val_s, test_s = (all_samples[:n_train],
                           all_samples[n_train:n_train+n_val],
                           all_samples[n_train+n_val:])

train_ds = RBCClsDataset(train_s, augment=True)
val_ds   = RBCClsDataset(val_s)
test_ds  = RBCClsDataset(test_s)

# Tính toán sample_weights dựa trên phân phối class thực tế của tập train
from collections import Counter
train_labels = [label for _, label in train_s]
class_counts = Counter(train_labels)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=(DEVICE=="cuda"))
val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))
print(f"Train/Val/Test: {n_train}/{n_val}/{n_test}")
"""),

md("## 🏗 Model Architectures\n### Pipeline A — Custom CNN Classifier"),

code("""\
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x); max_o, _ = torch.max(x, dim=1, keepdim=True); avg_o = torch.mean(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([max_o, avg_o], dim=1))

class CNNClassifier(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam1  = CBAM(32)
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam2  = CBAM(64)
        self.layer3 = nn.Sequential(nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2))
        self.cbam3  = CBAM(128)
        self.layer4 = nn.Sequential(nn.Conv2d(128,256,3, padding=1), nn.BatchNorm2d(256),nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.layer4(x)
        return self.head(x)

model_cnn = CNNClassifier(n_classes=len(CLASSES)).to(DEVICE)
print(f"CNN params: {sum(p.numel() for p in model_cnn.parameters()):,}")
"""),

md("### Pipeline B — Custom RNN (BiGRU) Classifier"),

code("""\
class BiGRUClassifier(nn.Module):
    \"\"\"Treat each row of the 80×80 image as a time step (seq_len=80, input=80*3=240).\"\"\"
    def __init__(self, n_classes=9, hidden=128, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(IMG_SIZE * 3, hidden)
        self.gru = nn.GRU(hidden, hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)  # B, H, W*C
        x = self.input_proj(x)                            # B, H, hidden
        out, _ = self.gru(x)                              # B, H, hidden*2
        return self.head(out[:, -1, :])                   # last timestep

model_rnn = BiGRUClassifier(n_classes=len(CLASSES)).to(DEVICE)
print(f"BiGRU params: {sum(p.numel() for p in model_rnn.parameters()):,}")
"""),

md("### Pipeline C — Custom Vision Transformer (ViT) Classifier"),

code("""\
class TransBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=3, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim*mlp_ratio, dim), nn.Dropout(drop))
    def forward(self, x):
        n = self.norm1(x); x = x + self.attn(n,n,n)[0]
        return x + self.mlp(self.norm2(x))

class ViTClassifier(nn.Module):
    def __init__(self, img_size=80, patch_size=8, in_ch=3, n_classes=9,
                 embed_dim=128, depth=4, heads=4):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed  = nn.Parameter(torch.randn(1, n_patches + 1, embed_dim) * 0.02)
        self.blocks     = nn.Sequential(*[TransBlock(embed_dim, heads) for _ in range(depth)])
        self.norm       = nn.LayerNorm(embed_dim)
        self.head       = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_proj(x).flatten(2).transpose(1,2)   # B, N, E
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed  # B, N+1, E
        x   = self.norm(self.blocks(x))
        return self.head(x[:, 0])                           # CLS token

model_vit = ViTClassifier(n_classes=len(CLASSES)).to(DEVICE)
print(f"ViT params:  {sum(p.numel() for p in model_vit.parameters()):,}")
"""),

md("## 🏋 Training"),

code("""\
def train_cls(model, name, epochs=EPOCHS, lr=LR):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Dùng ReduceLROnPlateau thay cho Cosine để adaptive hơn với Early Stopping
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5)
    # Kỹ thuật: Label Smoothing (0.1) giúp model chống overfitting và tự tin thái quá
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    early_stop_patience = 10
    min_delta = 0.003
    epochs_no_improve = 0

    best_acc, best_state = 0., None
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        sched.step(acc)

        history["train_loss"].append(running / len(train_dl))
        history["val_acc"].append(acc)

        if acc > best_acc + min_delta:
            best_acc  = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if True:  # Log every epoch as requested
            curr_lr = opt.param_groups[0]['lr']
            print(f"[{name}] Epoch {epoch:3d}/{epochs} | LR {curr_lr:.1e} | "
                  f"Loss {running/len(train_dl):.4f} | Val Acc {acc:.4f}")
                  
        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping tại epoch {epoch} (Không cải thiện sau {early_stop_patience} epochs)")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{RESULTS_DIR}/{name}_best.pt")
    print(f"✅ {name} best Val Acc = {best_acc:.4f}")
    return history
"""),

code("""\
t0=time.time(); hist_cnn = train_cls(model_cnn, "CNN_Cls")
print(f"⏱ {(time.time()-t0)/60:.1f} min\\n")
t0=time.time(); hist_rnn = train_cls(model_rnn, "RNN_Cls")
print(f"⏱ {(time.time()-t0)/60:.1f} min\\n")
t0=time.time(); hist_vit = train_cls(model_vit, "ViT_Cls")
print(f"⏱ {(time.time()-t0)/60:.1f} min")
"""),

md("## 📊 Evaluation"),

code("""\
def evaluate_cls(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)

results_cls = {}
for m, name in [(model_cnn,"CNN"), (model_rnn,"RNN"), (model_vit,"ViT")]:
    preds, labels = evaluate_cls(m, test_dl)
    report = classification_report(labels, preds, target_names=CLASSES, output_dict=True)
    results_cls[name] = {
        "Accuracy": report["accuracy"],
        "Macro F1": report["macro avg"]["f1-score"],
        "preds": preds, "labels": labels,
    }
    print(f"\\n{'='*50}")
    print(f"  {name} — Accuracy: {report['accuracy']:.4f}  Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(classification_report(labels, preds, target_names=CLASSES))
"""),

code("""\
# Confusion matrices & comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, r) in zip(axes, results_cls.items()):
    cm = confusion_matrix(r["labels"], r["preds"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax.set_title(f"{name}\\nAcc={r['Accuracy']:.3f} F1={r['Macro F1']:.3f}", fontsize=10)
    ax.tick_params(axis='x', rotation=45)
plt.suptitle("Confusion Matrices: CNN vs RNN vs Transformer", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150)
plt.show()

names  = list(results_cls.keys())
accs   = [results_cls[n]["Accuracy"] for n in names]
f1s    = [results_cls[n]["Macro F1"] for n in names]
fig, ax = plt.subplots(figsize=(7,4))
x = np.arange(len(names)); w = 0.35
ax.bar(x-w/2, accs, w, label="Accuracy", color=["steelblue","coral","seagreen"])
ax.bar(x+w/2, f1s,  w, label="Macro F1", color=["steelblue","coral","seagreen"], alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0,1); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Test Set: Accuracy & Macro F1", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png", dpi=150)
plt.show()
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — END-TO-END PIPELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

E2E_CELLS = [

md("# 🔗 Notebook 3: End-to-End Pipeline Comparison\n"
   "Combine Segmentation + Classification models into 3 full pipelines and compare results.\n"
   "> **Run Notebooks 1 & 2 first** to generate the saved model weights."),

code("""\
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_BASE = r"c:\\Users\\DELL\\Desktop\\Vinh Hoang\\Master Program\\Học sâu\\Project\\Dataset"
# DATASET_BASE = "/content/drive/MyDrive/Project/Dataset"

SEG_RESULTS = os.path.join(os.path.dirname(DATASET_BASE), "results", "segmentation")
CLS_RESULTS = os.path.join(os.path.dirname(DATASET_BASE), "results", "classification")
cls_dir     = os.path.join(DATASET_BASE, "Elsafty_RBCs_for_Classification", "Cropped images")
# CHỈ lấy những thư mục thực sự chứa ảnh .png (giống như Notebook 2 đã làm)
CLASSES = sorted([d.name for d in Path(cls_dir).iterdir() if d.is_dir() and any(d.glob("*.png"))])
IMG_SIZE = 80
print(f"Device: {DEVICE}")
print(f"Classes ({len(CLASSES)}): {CLASSES}")
"""),

code("""\
# Re-define model classes (copy architecture from Notebooks 1 & 2)
# ── Seg models ──
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(1, channels//reduction), 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(max(1, channels//reduction), channels, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
    def forward(self, x):
        x = x * self.ca(x); max_o, _ = torch.max(x, dim=1, keepdim=True); avg_o = torch.mean(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([max_o, avg_o], dim=1))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.cbam = CBAM(out_ch)
    def forward(self,x): return self.cbam(self.net(x))

class MiniUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[16,32,64]):
        super().__init__()
        self.encoders=nn.ModuleList(); self.pools=nn.ModuleList()
        self.decoders=nn.ModuleList(); self.upconvs=nn.ModuleList()
        prev=in_ch
        for f in features:
            self.encoders.append(DoubleConv(prev,f)); self.pools.append(nn.MaxPool2d(2)); prev=f
        self.bottleneck=DoubleConv(prev,prev*2); prev=prev*2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev,f,2,2))
            self.decoders.append(DoubleConv(f*2,f)); prev=f
        self.head=nn.Conv2d(prev,out_ch,1)
    def forward(self,x):
        skips=[]
        for enc,pool in zip(self.encoders,self.pools):
            x=enc(x); skips.append(x); x=pool(x)
        x=self.bottleneck(x); skips=skips[::-1]
        for up,dec,skip in zip(self.upconvs,self.decoders,skips):
            x=up(x)
            if x.shape!=skip.shape: x=F.interpolate(x,size=skip.shape[2:])
            x=dec(torch.cat([skip,x],dim=1))
        return self.head(x)

# ── Cls models ──
class CNNClassifier(nn.Module):
    def __init__(self,n=9):
        super().__init__()
        self.layer1=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.cbam1=CBAM(32)
        self.layer2=nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.cbam2=CBAM(64)
        self.layer3=nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.cbam3=CBAM(128)
        self.layer4=nn.Sequential(nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(0.4),nn.Linear(256,n))
    def forward(self,x): return self.head(self.layer4(self.cbam3(self.layer3(self.cbam2(self.layer2(self.cbam1(self.layer1(x))))))))

print("Model classes defined ✅")
"""),

code("""\
# ── Load saved weights ────────────────────────────────────────────────────────
def load_model(cls, path, device=DEVICE):
    m = cls().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

seg_cnn = load_model(MiniUNet, f"{SEG_RESULTS}/CNN_UNet_best.pt")
cls_cnn = load_model(CNNClassifier, f"{CLS_RESULTS}/CNN_Cls_best.pt")
print("Loaded CNN pipeline ✅")
# Add RNN & ViT models here similarly after pasting architectures from NB1 & NB2
"""),

code("""\
# ── Pipeline inference function ───────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def run_pipeline(seg_model, cls_model, crop_img_pil):
    \"\"\"
    Given an 80x80 PIL crop:
      1. Segmentation → binary mask
      2. Apply mask → segmented cell
      3. Classification → class probabilities
    \"\"\"
    tensor = img_tf(crop_img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask_logit = seg_model(tensor)
        mask       = (torch.sigmoid(mask_logit) > 0.5).float()
        logits     = cls_model(tensor)
        probs      = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_cls   = CLASSES[probs.argmax()]
    confidence = probs.max()
    mask_np    = mask.squeeze().cpu().numpy()
    return pred_cls, confidence, mask_np, probs


def compare_pipelines(pipelines, test_img_paths, n=5):
    \"\"\"Show results of all pipelines on n sample images.\"\"\"
    samples = test_img_paths[:n]
    n_pipes = len(pipelines)
    fig, axes = plt.subplots(n, 1 + n_pipes, figsize=(4 * (1+n_pipes), 3.5 * n))

    for i, path in enumerate(samples):
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        axes[i][0].imshow(img); axes[i][0].set_title("Input"); axes[i][0].axis("off")

        for j, (name, seg, cls) in enumerate(pipelines):
            pred_cls, conf, mask, probs = run_pipeline(seg, cls, img)
            ax = axes[i][1+j]
            ax.imshow(mask, cmap="gray")
            ax.set_title(f"{name}\\n{pred_cls}\\n({conf:.0%})", fontsize=8)
            ax.axis("off")

    plt.suptitle("End-to-End Pipeline Comparison: CNN vs RNN vs Transformer",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(DATASET_BASE), "results", "e2e_comparison.png"), dpi=150)
    plt.show()

print("Pipeline functions defined ✅")
"""),

code("""\
# ── Run comparison on test crops ──────────────────────────────────────────────
test_crops = sorted(glob.glob(
    os.path.join(DATASET_BASE, "Elsafty_RBCs_for_Classification", "Cropped images", "**", "*.png"),
    recursive=True
))[:50]  # use first 50 as demo images

pipelines = [
    ("CNN", seg_cnn, cls_cnn),
    # ("RNN", seg_rnn, cls_rnn),   # uncomment after loading RNN pipeline
    # ("ViT", seg_vit, cls_vit),   # uncomment after loading ViT pipeline
]

compare_pipelines(pipelines, test_crops, n=5)
"""),

code("""\
# ── Final summary table ───────────────────────────────────────────────────────
import pandas as pd

summary = {
    "Pipeline": ["CNN", "RNN", "Transformer"],
    "Seg Model": ["Mini U-Net", "ConvLSTM U-Net", "ViT-UNet"],
    "Cls Model": ["Custom CNN", "Bi-GRU", "Custom ViT"],
    # Fill in from Notebooks 1 & 2 results
    "Seg Dice":  [None, None, None],
    "Seg IoU":   [None, None, None],
    "Cls Acc":   [None, None, None],
    "Cls F1":    [None, None, None],
}

df = pd.DataFrame(summary)
print(df.to_string(index=False))
# After filling in metrics:
# df.to_csv(os.path.join(os.path.dirname(DATASET_BASE), "results", "final_summary.csv"), index=False)
"""),
]

# ══════════════════════════════════════════════════════════════════════════════
# SMART GENERATE NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════════

targets = {
    "1": ("01_Segmentation_Comparison.ipynb", SEG_CELLS),
    "2": ("02_Classification_Comparison.ipynb", CLS_CELLS),
    "3": ("03_EndToEnd_Comparison.ipynb", E2E_CELLS),
}

requested = [arg for arg in sys.argv[1:] if arg in targets]

if not requested:
    print("\n🧐 Không có tham số chọn lọc. Đang kiểm tra để sinh tất cả các Notebook...")
    for key, (filename, cells) in targets.items():
        if os.path.exists(filename):
            print(f"⚠️  Bỏ qua {filename}: File đã tồn tại (để bảo vệ log training).")
            print(f"   Dùng 'python generate_notebooks.py {key}' nếu bạn muốn ghi đè file này.")
        else:
            save(notebook(cells), filename)
            print(f"✅  Created: {filename}")
else:
    print(f"\n🚀 Đang sinh các Notebook được yêu cầu: {requested}")
    for key in requested:
        filename, cells = targets[key]
        save(notebook(cells), filename)
        print(f"✅  Updated (Forced): {filename}")

print("\n🎉 Xong! Hãy mở các file trên trong VS Code.")
