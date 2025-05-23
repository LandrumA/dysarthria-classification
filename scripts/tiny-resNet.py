#!/usr/bin/env python3
"""
Speaker-exclusive 2-D MFCC ResNet for TORGO
‒ adds sklearn classification report + loss curve plot.
"""

import os, re, glob
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt   # for loss curve

# ── CONFIG ──────────────────────────────────────────────────────────────
ROOT_DIR   = "/home/the_fat_cat/Documents/data/features/MFCCs/TORGO"
TEST_SIZE  = 0.20
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_FILE  = "loss_curve.png"     # saved to CWD
# ────────────────────────────────────────────────────────────────────────


# ── FILE GATHERING ──────────────────────────────────────────────────────
paths, labels, speakers = [], [], []
pat = re.compile(r"TORGO_[mf]_(?P<spk>[A-Z]{1,2}\d{2})_\d+_(?P<cls>[ac])\.npy$", re.I)

for fp in glob.iglob(os.path.join(ROOT_DIR, "**", "*.npy"), recursive=True):
    m = pat.search(os.path.basename(fp))
    if not m:
        continue
    speakers.append(m.group("spk"))
    labels.append(1 if m.group("cls").lower() == "a" else 0)
    paths.append(fp)

paths, labels, speakers = np.array(paths), np.array(labels), np.array(speakers)
print(f"Loaded {len(paths)} files  –  afflicted {labels.sum()}  control {len(labels)-labels.sum()}")

# ── SPEAKER-EXCLUSIVE SPLIT ─────────────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
train_idx, test_idx = next(gss.split(paths, labels, groups=speakers))

assert not set(speakers[train_idx]) & set(speakers[test_idx]), "Speaker leakage!"

# ── DATASETS ────────────────────────────────────────────────────────────
class MFCC2D(Dataset):
    def __init__(self, fpaths, ys):
        self.fpaths = fpaths
        self.ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        sample = np.load(self.fpaths[0])
        self.H, self.W = sample.shape

    def __len__(self): return len(self.fpaths)

    def __getitem__(self, i):
        arr = np.load(self.fpaths[i]).astype(np.float32)
        if arr.shape != (self.H, self.W):
            pad = np.zeros((self.H, self.W), dtype=np.float32)
            h, w = min(self.H, arr.shape[0]), min(self.W, arr.shape[1])
            pad[:h, :w] = arr[:h, :w]
            arr = pad
        return torch.from_numpy(arr).unsqueeze(0), self.ys[i]

train_loader = DataLoader(MFCC2D(paths[train_idx], labels[train_idx]), BATCH_SIZE, True,  num_workers=4)
test_loader  = DataLoader(MFCC2D(paths[test_idx],  labels[test_idx]),  BATCH_SIZE, False, num_workers=4)

# ── MODEL ───────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, ch): super().__init__(); self.c1=nn.Conv2d(ch,ch,3,1,1); self.b1=nn.BatchNorm2d(ch); self.c2=nn.Conv2d(ch,ch,3,1,1); self.b2=nn.BatchNorm2d(ch)
    def forward(self,x): return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+x)

class TinyResNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch,base,3,1,1), nn.BatchNorm2d(base), nn.ReLU(), nn.MaxPool2d(2))
        self.layer1 = Block(base)
        self.layer2 = nn.Sequential(nn.Conv2d(base,base*2,3,2,1), nn.BatchNorm2d(base*2), nn.ReLU(), Block(base*2))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base*2,1))
    def forward(self,x): x=self.stem(x); x=self.layer1(x); x=self.layer2(x); return self.head(x)   # logits

model = TinyResNet().to(DEVICE)
crit  = nn.BCEWithLogitsLoss()
opt   = torch.optim.Adam(model.parameters(), lr=LR)

# ── TRAINING LOOP ───────────────────────────────────────────────────────
train_losses = []

for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * yb.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch:02}/{EPOCHS} – loss {epoch_loss:.4f}")

# ── LOSS CURVE ──────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.plot(range(1, EPOCHS+1), train_losses, marker="o")
plt.xlabel("Epoch"), plt.ylabel("Training loss"), plt.title("Loss curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE)
print(f"Loss curve saved to {PLOT_FILE}")

# ── EVALUATION & CLASSIFICATION REPORT ──────────────────────────────────
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb).cpu()
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).float()
        y_true.extend(yb.squeeze(1).tolist())
        y_pred.extend(preds.squeeze(1).tolist())

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["control","afflicted"], digits=4))