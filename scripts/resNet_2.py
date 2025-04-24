#!/usr/bin/env python3
"""
Speaker-exclusive 2-D MFCC ResNet for TORGO
‒ z-score normalisation (µ, σ from training set)
‒ per-epoch logging to /results/resNet_logs/*.txt
"""

import os, re, glob, datetime, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
import random

# ── CONFIG ──────────────────────────────────────────
ROOT_DIR   = "/Users/visar/Downloads/features/MFCCs/TORGO"
LOG_DIR    = "/Users/visar/Desktop/S25/Classes/Deep Learning/results"
MODEL_DIR  = "/Users/visar/Desktop/S25/Classes/Deep Learning/models"
CURVE_DIR  = "/Users/visar/Desktop/S25/Classes/Deep Learning/results/resNet_loss-curves"
TEST_SIZE  = 0.20
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PLOT_FILE  = "loss_curve.png"
NUM_WORKERS = os.cpu_count() // 2

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

# ── DATASET CLASS (moved outside main) ─────────────────────────
class MFCC2D(Dataset):
    def __init__(self, fpaths, ys, mu, sigma, H, W):
        self.fpaths = fpaths
        self.ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        self.mu, self.sigma = mu.astype(np.float32), sigma.astype(np.float32)
        self.H, self.W = H, W

    def __len__(self): return len(self.fpaths)

    def __getitem__(self, i):
        x = np.load(self.fpaths[i]).astype(np.float32)
        if x.shape != (self.H, self.W):
            pad = np.zeros((self.H, self.W), dtype=np.float32)
            h, w = min(self.H, x.shape[0]), min(self.W, x.shape[1])
            pad[:h, :w] = x[:h, :w]
            x = pad
        x = (x - self.mu) / self.sigma
        return torch.from_numpy(x).unsqueeze(0).float(), self.ys[i]

# ── FILE GATHERING ──────────────────────────────────────
pat = re.compile(r"TORGO_[mf]_(?P<spk>[A-Z]{1,2}\d{2})_\d+_(?P<cls>[ac])\.npy$", re.I)

all_paths, all_labels, all_speakers = [], [], []
for fp in glob.iglob(os.path.join(ROOT_DIR, "**", "*.npy"), recursive=True):
    m = pat.search(os.path.basename(fp))
    if not m:
        continue
    all_paths.append(fp)
    all_labels.append(1 if m.group("cls").lower() == "a" else 0)
    all_speakers.append(m.group("spk"))

all_paths    = np.array(all_paths)
all_labels   = np.array(all_labels)
all_speakers = np.array(all_speakers)

random.seed(42)
aff_idx  = np.where(all_labels == 1)[0]
ctrl_idx = np.where(all_labels == 0)[0]

if len(aff_idx) < 2000 or len(ctrl_idx) < 2000:
    raise ValueError("Not enough samples to draw 2 000 from each class.")

sel_aff  = random.sample(list(aff_idx),  2000)
sel_ctrl = random.sample(list(ctrl_idx), 2000)
sel_idx  = np.array(sel_aff + sel_ctrl)

paths    = all_paths[sel_idx]
labels   = all_labels[sel_idx]
speakers = all_speakers[sel_idx]

print(f"Balanced dataset  –  afflicted {labels.sum()}  control {len(labels)-labels.sum()}  total {len(labels)}")

# ── SPLIT ──

if __name__ == "__main__":
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(paths, labels, groups=speakers))
    assert not set(speakers[train_idx]) & set(speakers[test_idx]), "Speaker leakage!"

    sample = np.load(paths[train_idx][0]).astype(np.float32)
    H, W = sample.shape
    sums = np.zeros((H, W), dtype=np.float32)
    sq_sums = np.zeros((H, W), dtype=np.float32)
    count = 0
    for fp in paths[train_idx]:
        x = np.load(fp).astype(np.float32)
        if x.shape != (H, W):
            pad = np.zeros((H, W), dtype=np.float32)
            h, w = min(H, x.shape[0]), min(W, x.shape[1])
            pad[:h, :w] = x[:h, :w]
            x = pad
        sums += x
        sq_sums += x ** 2
        count += 1

    mu = sums / count
    sigma = np.sqrt(np.maximum(sq_sums / count - mu ** 2, 1e-7)).astype(np.float32)

    train_loader = DataLoader(MFCC2D(paths[train_idx], labels[train_idx], mu, sigma, H, W), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(MFCC2D(paths[test_idx],  labels[test_idx],  mu, sigma, H, W), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class Block(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
            self.b2 = nn.BatchNorm2d(ch)
        def forward(self, x):
            return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + x)

    class TinyResNet(nn.Module):
        def __init__(self, in_ch=1, base=16):
            super().__init__()
            self.stem  = nn.Sequential(
                nn.Conv2d(in_ch, base, 3, 1, 1),
                nn.BatchNorm2d(base), nn.ReLU(), nn.MaxPool2d(2))
            self.layer1 = Block(base)
            self.layer2 = nn.Sequential(
                nn.Conv2d(base, base*2, 3, 2, 1),
                nn.BatchNorm2d(base*2), nn.ReLU(), Block(base*2))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Flatten(), nn.Linear(base*2, 1))
        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            return self.head(x)

    model = TinyResNet().to(DEVICE)
    crit  = nn.BCEWithLogitsLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"resnet_{stamp}.txt")
    with open(log_path, "w") as lf:
        lf.write("Epoch\tTrainLoss\n")

    train_losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * yb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch:02}/{EPOCHS} – loss {epoch_loss:.4f}")
        with open(log_path, "a") as lf:
            lf.write(f"{epoch}\t{epoch_loss:.6f}\n")

    MODEL_FILE = os.path.join(MODEL_DIR, "resNet_2.pth")
    CURVE_FILE = os.path.join(CURVE_DIR, "resNet_2.png")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"Model weights saved to {MODEL_FILE}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Loss curve")
    plt.grid(True); plt.tight_layout(); plt.savefig(CURVE_FILE)
    print(f"Loss-curve figure saved to {CURVE_FILE}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).cpu()
            preds = (probs > 0.5).float()
            y_true.extend(yb.squeeze(1).tolist())
            y_pred.extend(preds.squeeze(1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["control", "afflicted"], digits=4)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", cr)
    with open(log_path, "a") as lf:
        lf.write("\nConfusion matrix:\n")
        np.savetxt(lf, cm, fmt="%d")
        lf.write("\n\n")
        lf.write(cr)
    print(f"Log written to {log_path}")