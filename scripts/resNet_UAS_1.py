#!/usr/bin/env python3
"""
ResNet-18 for UASpeech (3-s MFCC «images»)
──────────────────────────────────────────
• Input = (1, 39, 300) tensor from 2-D MFCC .npy files
• Binary label: _a → 1 (afflicted)   _c → 0 (control)
• Speaker-independent 60 / 20 / 20 split via GroupShuffleSplit
  (group = *entire* base filename, per user instruction)
• Class-balanced WeightedRandomSampler
• Metrics: loss curve, accuracy, ROC-AUC, confusion matrix
• GPU auto-detect
"""

# ── imports ─────────────────────────────────────────────────────────────
import os, glob, math, random, json, time
import numpy as np
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ── CONFIG ──────────────────────────────────────────────────────────────
ROOT_2D   = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"
OUT_DIR   = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/models/UASpeech/resNet"
EPOCHS    = 25
BATCH     = 32
LR        = 1e-3
MAX_LEN   = 300        # 3.0 s → 300 frames (10 ms hop)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42
random.seed(SEED);  np.random.seed(SEED);  torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────
def pad_trunc(mfcc: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """Pad with zeros (right) or truncate to [39, max_len]."""
    if mfcc.shape[1] >= max_len:
        return mfcc[:, :max_len]
    pad = np.zeros((mfcc.shape[0], max_len - mfcc.shape[1]), dtype=mfcc.dtype)
    return np.concatenate([mfcc, pad], axis=1)

def label_from_fname(fname: str) -> int:
    return 1 if fname.rstrip(".npy").split("_")[-1] == "a" else 0   # _a afflicted, _c control

# ── dataset ─────────────────────────────────────────────────────────────
class UASpeechMFCC(Dataset):
    def __init__(self, files, transform=None):
        self.files      = files
        self.transform  = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fpath   = self.files[idx]
        base    = os.path.splitext(os.path.basename(fpath))[0]
        label   = label_from_fname(base)
        mfcc    = np.load(fpath)                # (39, T)
        mfcc    = pad_trunc(mfcc)               # (39, 300)
        # simple per-clip standardization
        mfcc    = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        tensor  = torch.from_numpy(mfcc).float().unsqueeze(0)  # (1, 39, 300)
        return tensor, label

# ── collect files & speaker IDs ─────────────────────────────────────────
all_files = sorted(glob.glob(os.path.join(ROOT_2D, "*.npy")))
groups    = [os.path.splitext(os.path.basename(f))[0] for f in all_files]   # ← ENTIRE basename

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
train_idx, temp_idx = next(gss1.split(all_files, groups=groups))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx   = next(gss2.split([all_files[i] for i in temp_idx],
                                      groups=[groups[i] for i in temp_idx]))

train_files = [all_files[i] for i in train_idx]
val_files   = [all_files[temp_idx[i]] for i in val_idx]
test_files  = [all_files[temp_idx[i]] for i in test_idx]

print(f"Files  | train {len(train_files)}  val {len(val_files)}  test {len(test_files)}")

# ── data loaders with class balance ─────────────────────────────────────
def make_loader(file_list, shuffle=True):
    labels = [label_from_fname(os.path.basename(f)) for f in file_list]
    if shuffle:
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(UASpeechMFCC(file_list), batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        return DataLoader(UASpeechMFCC(file_list), batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

train_loader = make_loader(train_files, shuffle=True)
val_loader   = make_loader(val_files,   shuffle=False)
test_loader  = make_loader(test_files,  shuffle=False)

# ── model ───────────────────────────────────────────────────────────────
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1-channel
model.fc    = nn.Linear(model.fc.in_features, 2)  # binary
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.3)

# ── training loop ───────────────────────────────────────────────────────
def epoch_pass(loader, train=True):
    model.train(train)
    torch.set_grad_enabled(train)
    running_loss, preds, labels = 0., [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        if train:
            optimizer.zero_grad();  loss.backward();  optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    return running_loss / len(loader.dataset), acc

best_val_acc, best_state = 0., None
history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

for epoch in range(1, EPOCHS + 1):
    tl, ta = epoch_pass(train_loader, train=True)
    vl, va = epoch_pass(val_loader,   train=False)
    scheduler.step(vl)
    history["train_loss"].append(tl);  history["val_loss"].append(vl)
    history["train_acc"].append(ta);   history["val_acc"].append(va)
    print(f"[{epoch:02d}/{EPOCHS}]  loss: {tl:.4f}/{vl:.4f}  acc: {ta:.3f}/{va:.3f}")
    if va > best_val_acc:
        best_val_acc = va
        best_state   = model.state_dict().copy()

# ── evaluation on test set ──────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval();  preds, probs, labels = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        probs.extend(F.softmax(out, dim=1)[:,1].cpu().numpy())
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        labels.extend(y.numpy())

test_acc  = accuracy_score(labels, preds)
test_auc  = roc_auc_score(labels, probs)
cm        = confusion_matrix(labels, preds)
print("\nTEST  acc {:.3f}  AUC {:.3f}".format(test_acc, test_auc))
print("Confusion matrix:\n", cm)
print("\nClassification report:\n", classification_report(labels, preds, digits=3))

# ── save artifacts ──────────────────────────────────────────────────────
torch.save(best_state, os.path.join(OUT_DIR, "resnet18_best.pt"))
with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)
print("\nModel + training history saved to", OUT_DIR)
