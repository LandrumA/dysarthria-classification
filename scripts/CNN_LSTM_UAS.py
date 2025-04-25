#!/usr/bin/env python3
"""
CNN-BiLSTM Dysarthria Classifier — UASpeech 2-D MFCC
────────────────────────────────────────────────────
• Input  : *.npy (39×T) MFCC files in DATA_DIR
• Split  : 5-fold Stratified Group K-Fold (groups = speakers, no leakage)
• Model  : ResNet-18 front-end → 2-layer BiLSTM → 2-class head
• Augment: Gaussian noise + SpecAugment time/freq masking (train only)
• Metrics: per-fold   acc · ROC-AUC (if both classes) · macro-F1
           pooled     acc · ROC-AUC · macro-F1 · confusion matrix
• Output : results/UASpeech/curves/roc_stratkfold.png
           results/UASpeech/logs/stratkfold_report.txt
"""

# ── configuration ───────────────────────────────────────────────────────
DATA_DIR      = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"
RESULT_ROOT   = "results/UASpeech"
N_SPLITS      = 5
EPOCHS        = 12
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
SEED          = 42

# ── imports ─────────────────────────────────────────────────────────────
import os, glob, random
import numpy as np
from collections import defaultdict

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── reproducibility ─────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── helper functions ────────────────────────────────────────────────────
def speaker_id(path: str) -> str:
    b = os.path.basename(path).replace(".npy", "").split("_")
    return "_".join([b[0], b[1], b[2], b[4]])        # UAS_m_13_a

def cmvn(m: np.ndarray) -> np.ndarray:
    mu = m.mean(1, keepdims=True)
    sd = m.std(1, keepdims=True) + 1e-9
    return (m - mu) / sd

def pad_or_trunc(m: np.ndarray, T: int = 300) -> np.ndarray:
    return m[:, :T] if m.shape[1] >= T else np.pad(m, ((0, 0), (0, T - m.shape[1])))

class Augment:
    def __init__(self, sigma=0.01, t_mask=20, f_mask=4):
        self.sigma, self.t_mask, self.f_mask = sigma, t_mask, f_mask
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x += torch.randn_like(x) * self.sigma
        _, F, T = x.shape
        t0 = random.randint(0, max(0, T - self.t_mask))
        x[:, :, t0:t0 + self.t_mask] = 0
        f0 = random.randint(0, max(0, F - self.f_mask))
        x[:, f0:f0 + self.f_mask, :] = 0
        return x
AUGMENT = Augment()

# ── dataset ─────────────────────────────────────────────────────────────
class MFCCDataset(Dataset):
    def __init__(self, paths, augment=False):
        self.paths   = paths
        self.augment = augment
        self.labels  = [1 if os.path.basename(p).split("_")[4] == "a" else 0
                        for p in paths]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        m = np.load(self.paths[idx])
        m = pad_or_trunc(cmvn(m))
        x = torch.from_numpy(m).unsqueeze(0).float()   # (1,39,300)
        if self.augment:
            x = AUGMENT(x)
        return x, self.labels[idx]

# ── model ───────────────────────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
    def __init__(self, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = resnet18(weights=None)
        self.cnn.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.cnn.fc    = nn.Identity()                 # (B,512)
        self.lstm = nn.LSTM(input_size=512,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, 2)
    def forward(self, x):                              # x:(B,1,39,300)
        feat = self.cnn(x)                             # (B,512)
        seq  = feat.unsqueeze(1)                       # (B,1,512)
        out, _ = self.lstm(seq)                        # (B,1,2H)
        return self.fc(out[:, -1])                     # (B,2)

# ── load / index data ───────────────────────────────────────────────────
spk2paths = defaultdict(list)
for p in tqdm(glob.glob(os.path.join(DATA_DIR, "*.npy")), desc="Indexing"):
    m = np.load(p)
    if m.std() < 1e-5:      # skip near-silence
        continue
    spk2paths[speaker_id(p)].append(p)

paths, labels, groups = [], [], []
for sid, pl in spk2paths.items():
    paths.extend(pl)
    labels.extend([1 if sid.endswith("_a") else 0] * len(pl))
    groups.extend([sid] * len(pl))

print(f"Total clips: {len(paths)} | Speakers: {len(spk2paths)}")

# ── splitter ───────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

fold_metrics = []
all_y, all_p = [], []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(paths, labels, groups), 1):
    print(f"\n── Fold {fold}/{N_SPLITS} ──")

    train_paths = [paths[i] for i in train_idx]
    val_paths   = [paths[i] for i in val_idx]

    ds_train = MFCCDataset(train_paths, augment=True)
    ds_val   = MFCCDataset(val_paths,   augment=False)

    # balanced sampler (if both classes present)
    n_pos = sum(ds_train.labels)
    n_neg = len(ds_train) - n_pos
    if n_pos > 0 and n_neg > 0:
        cls_w = torch.tensor([len(ds_train) / n_pos,
                              len(ds_train) / n_neg], dtype=torch.float)
        samp_w = cls_w[ds_train.labels]
        sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = True
        print("  ⚠️  One class missing in training split – using shuffled loader.")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          sampler=sampler, shuffle=shuffle_flag,
                          num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

    net   = CNN_BiLSTM().to(DEVICE)
    opt   = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()

    # training loop
    net.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        for X, y in dl_train:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = lossf(net(X), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * X.size(0)
        print(f"  ep {ep:02d}/{EPOCHS}  loss={total_loss / len(ds_train):.4f}")

    # validation
    net.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for X, y in dl_val:
            probs = torch.softmax(net(X.to(DEVICE)), 1)[:, 1].cpu().numpy()
            y_true.extend(y.numpy())
            y_prob.extend(probs)

    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    f1  = f1_score(y_true, y_pred, average="macro")
    fold_metrics.append((acc, auc, f1))
    print(f"  ► acc={acc:.4f}  auc={auc}  macro-F1={f1:.4f}")

    all_y.extend(y_true)
    all_p.extend(y_prob)

# ── pooled results ──────────────────────────────────────────────────────
pooled_pred = (np.array(all_p) >= 0.5).astype(int)
pooled_acc  = accuracy_score(all_y, pooled_pred)
pooled_auc  = roc_auc_score(all_y, all_p)
pooled_f1   = f1_score(all_y, pooled_pred, average="macro")
cmatrix     = confusion_matrix(all_y, pooled_pred)
cls_report  = classification_report(all_y, pooled_pred, digits=4)

print("\n==== POOLED RESULTS ====")
print(f"acc={pooled_acc:.4f}  auc={pooled_auc:.4f}  macro-F1={pooled_f1:.4f}")
print("Confusion matrix:\n", cmatrix)

# ── save artefacts ──────────────────────────────────────────────────────
curve_dir = os.path.join(RESULT_ROOT, "curves")
log_dir   = os.path.join(RESULT_ROOT, "logs")
os.makedirs(curve_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

RocCurveDisplay.from_predictions(all_y, all_p)
plt.title("UASpeech – Stratified-Group 5-Fold ROC")
plt.savefig(os.path.join(curve_dir, "roc_stratkfold.png"),
            dpi=300, bbox_inches="tight")
plt.close()

with open(os.path.join(log_dir, "stratkfold_report.txt"), "w") as f:
    f.write("UASpeech Stratified-Group 5-Fold Report\n")
    f.write("────────────────────────────────────────\n")
    for i, (a, u, ff) in enumerate(fold_metrics, 1):
        f.write(f"Fold {i}: acc={a:.4f}  auc={u}  macro-F1={ff:.4f}\n")
    f.write("\nPooled results\n")
    f.write(f"  acc      : {pooled_acc:.4f}\n")
    f.write(f"  ROC-AUC  : {pooled_auc:.4f}\n")
    f.write(f"  macro-F1 : {pooled_f1:.4f}\n\n")
    f.write("Classification report\n")
    f.write(cls_report)
    f.write("\nConfusion matrix\n")
    f.write(np.array2string(cmatrix))

print(f"\nSaved ROC → {os.path.join(curve_dir, 'roc_stratkfold.png')}")
print(f"Saved log → {os.path.join(log_dir, 'stratkfold_report.txt')}")
print("Done.")
