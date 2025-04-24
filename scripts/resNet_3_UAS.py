#!/usr/bin/env python3
"""
Speaker-exclusive 2-D MFCC ResNet for UASpeech
‒ balance to 4 700 afflicted / 4 700 control  
‒ speaker-exclusive 60/20/20 split  
‒ z-score normalisation (µ, σ from training speakers)  
‒ SpecAugment, ReduceLROnPlateau, early-stopping  
‒ logs, best-model checkpoint, loss-curve, test metrics
"""

import os, re, glob, random, datetime
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────
ROOT_DIR  = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech"
LOG_DIR   = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/results/resNet_logs"
MODEL_DIR = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/models"
CURVE_DIR = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/results/resNet_loss-curves"

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 2e-4
VAL_P      = 0.20
TEST_P     = 0.20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────

# ── FILE GATHERING & BALANCING ──────────────────────────────────────────
pat = re.compile(r"(?P<spk>[mf]_\d{2})_\d+_(?P<cls>[ac])\.npy$", re.I)

paths, labels, speakers = [], [], []
for fp in glob.iglob(os.path.join(ROOT_DIR, "**", "*.npy"), recursive=True):
    m = pat.search(os.path.basename(fp))
    if m:
        paths.append(fp)
        labels.append(1 if m.group("cls").lower() == "a" else 0)
        speakers.append(m.group("spk"))

paths, labels, speakers = map(np.array, (paths, labels, speakers))

random.seed(42)
aff_idx  = np.where(labels == 1)[0]
ctrl_idx = np.where(labels == 0)[0]
sel_aff  = random.sample(list(aff_idx),  4700)
sel_ctrl = random.sample(list(ctrl_idx), 4700)
keep_idx = np.array(sel_aff + sel_ctrl)

paths, labels, speakers = paths[keep_idx], labels[keep_idx], speakers[keep_idx]
print(f"Balanced dataset – afflicted {labels.sum()}  control {len(labels)-labels.sum()}  total {len(labels)}")

# ── SPEAKER-EXCLUSIVE 60/20/20 SPLIT ────────────────────────────────────
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_P, random_state=42)
train_val_idx, test_idx = next(gss.split(paths, labels, groups=speakers))

gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_P/(1-TEST_P), random_state=42)
train_idx, val_idx = next(gss_val.split(paths[train_val_idx],
                                        labels[train_val_idx],
                                        groups=speakers[train_val_idx]))
train_idx = train_val_idx[train_idx]
val_idx   = train_val_idx[val_idx]

print(f"Speakers – train {len(set(speakers[train_idx]))}  "
      f"val {len(set(speakers[val_idx]))}  "
      f"test {len(set(speakers[test_idx]))}")

# ── μ / σ COMPUTATION ON TRAIN SET ──────────────────────────────────────
sample = np.load(paths[train_idx][0]).astype(np.float32)
H, W = sample.shape
sums = np.zeros((H, W), np.float64)
sq_sums = np.zeros((H, W), np.float64)
for fp in paths[train_idx]:
    x = np.load(fp).astype(np.float32)
    if x.shape != (H, W):
        pad = np.zeros((H, W), np.float32)
        h, w = min(H, x.shape[0]), min(W, x.shape[1])
        pad[:h, :w] = x[:h, :w]
        x = pad
    sums += x
    sq_sums += x**2
mu = sums / len(train_idx)
sigma = np.sqrt(np.maximum(sq_sums/len(train_idx) - mu**2, 1e-7)).astype(np.float32)

# ── DATASET CLASS ───────────────────────────────────────────────────────
class MFCC2D(Dataset):
    def __init__(self, fpaths, ys, mu, sigma, H, W, augment=False):
        self.fpaths = fpaths
        self.ys = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        self.mu, self.sigma = mu, sigma
        self.H, self.W, self.augment = H, W, augment

    def __len__(self):
        return len(self.fpaths)

    def _specaugment(self, x):
        t = np.random.randint(0, 24); t0 = np.random.randint(0, self.W - t)
        x[:, t0:t0 + t] = 0
        f = np.random.randint(0, 6); f0 = np.random.randint(0, self.H - f)
        x[f0:f0 + f, :] = 0
        return x

    def __getitem__(self, i):
        x = np.load(self.fpaths[i]).astype(np.float32)
        if x.shape != (self.H, self.W):
            pad = np.zeros((self.H, self.W), np.float32)
            h, w = min(self.H, x.shape[0]), min(self.W, x.shape[1])
            pad[:h, :w] = x[:h, :w]
            x = pad
        if self.augment:
            x = self._specaugment(x)
        x = ((x - self.mu) / self.sigma).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), self.ys[i]

train_loader = DataLoader(
    MFCC2D(paths[train_idx], labels[train_idx], mu, sigma, H, W, augment=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(
    MFCC2D(paths[val_idx], labels[val_idx], mu, sigma, H, W),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(
    MFCC2D(paths[test_idx], labels[test_idx], mu, sigma, H, W),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── MODEL DEFINITION ────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + x)

class TinyResNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1), nn.BatchNorm2d(base),
            nn.ReLU(), nn.MaxPool2d(2))
        self.layer1 = Block(base)
        self.layer2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, 2, 1), nn.BatchNorm2d(base*2),
            nn.ReLU(), Block(base*2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base*2, 1))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)

model = TinyResNet().to(DEVICE)

# ── TRAINING PREP ───────────────────────────────────────────────────────
crit  = nn.BCEWithLogitsLoss()
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                   factor=0.5, patience=3)

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"resnet_uaspeech_{stamp}.txt")
with open(log_path, "w") as lf:
    lf.write("Epoch\tTrainLoss\tValLoss\n")

BEST_MODEL  = os.path.join(MODEL_DIR, "resNet_uaspeech_best.pth")
FINAL_MODEL = os.path.join(MODEL_DIR, "resNet_uaspeech.pth")
CURVE_FILE  = os.path.join(CURVE_DIR, "resNet_uaspeech.png")

best_val = np.inf
patience_left = 8
train_losses, val_losses = [], []

# ── TRAIN LOOP ──────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS + 1):
    model.train(); tloss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        tloss += loss.item() * yb.size(0)
    tloss /= len(train_loader.dataset)
    train_losses.append(tloss)

    model.eval(); vloss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            vloss += crit(model(xb), yb).item() * yb.size(0)
    vloss /= len(val_loader.dataset)
    val_losses.append(vloss)

    sched.step(vloss)
    print(f"Epoch {epoch:02}/{EPOCHS} – train {tloss:.4f}  val {vloss:.4f}")

    with open(log_path, "a") as lf:
        lf.write(f"{epoch}\t{tloss:.6f}\t{vloss:.6f}\n")

    if vloss < best_val - 1e-4:
        best_val = vloss; patience_left = 8
        torch.save(model.state_dict(), BEST_MODEL)
        print("  ↳ new best, model saved")
    else:
        patience_left -= 1
        if patience_left == 0:
            print("Early stop"); break

# ── AFTER TRAINING ──────────────────────────────────────────────────────
torch.save(model.state_dict(), FINAL_MODEL)
print("Final model saved")

plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses,   label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(CURVE_FILE)
print("Loss curve saved")

# ── TEST EVAL + AUC ─────────────────────────────────────────────────────
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()

y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        logits = model(xb).cpu()
        probs = torch.sigmoid(logits)
        y_true.extend(yb.squeeze(1).tolist())
        y_pred.extend((probs > 0.5).float().squeeze(1).tolist())
        y_prob.extend(probs.squeeze(1).tolist())

cm  = confusion_matrix(y_true, y_pred)
cr  = classification_report(y_true, y_pred,
                            target_names=["control", "afflicted"],
                            digits=4)
auc = roc_auc_score(y_true, y_prob)

print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", cr)
print(f"\nROC AUC: {auc:.4f}")

with open(log_path, "a") as lf:
    lf.write("\nConfusion matrix:\n")
    np.savetxt(lf, cm, fmt="%d")
    lf.write("\n\nClassification report:\n")
    lf.write(cr)
    lf.write(f"\n\nROC AUC: {auc:.6f}\n")

print(f"Log written to {log_path}")
