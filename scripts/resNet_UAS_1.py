#!/usr/bin/env python3
"""
Speaker-independent ResNet-18 for UASpeech MFCCs
────────────────────────────────────────────────
✓ Speaker ID = UAS_f_04_a  (recording index stripped)
✓ 15 afflicted, 13 control  →  28 total speakers
✓ Split: train 20, val 4 (2+2), test 4 (2+2)
✓ Full leakage audit
"""

# ── imports ─────────────────────────────────────────────────────────────
import os, glob, random, json, sys
from collections import defaultdict, Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

# ── CONFIG ──────────────────────────────────────────────────────────────
ROOT_2D = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"
OUT_DIR = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/models/UASpeech/resNet"
EPOCHS, BATCH, LR, MAX_LEN, PATIENCE = 25, 32, 1e-3, 300, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── helper functions ────────────────────────────────────────────────────
def speaker_id(fname: str) -> str:
    """Strip 4-digit recording index → UAS_f_04_a or UAS_m_13_c."""
    stem = fname.replace(".npy", "")
    p1, p2, p3, _, p5 = stem.split("_")   # UAS, f/m, 04, 1728, a/c
    return f"{p1}_{p2}_{p3}_{p5}"         # e.g. UAS_f_04_a

def pad_trunc(x: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    return x[:, :max_len] if x.shape[1] >= max_len \
           else np.pad(x, ((0,0),(0,max_len-x.shape[1])))

class MFCCDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        x = np.load(path)
        x = (pad_trunc(x) - x.mean()) / (x.std() + 1e-8)
        label = 1 if path.endswith("_a.npy") else 0
        return torch.from_numpy(x).float().unsqueeze(0), label

# ── build speaker dictionaries ─────────────────────────────────────────
speaker_files = defaultdict(list)
speaker_label = {}
for f in glob.glob(os.path.join(ROOT_2D, "*.npy")):
    spk = speaker_id(os.path.basename(f))
    lbl = 1 if spk.endswith("_a") else 0
    speaker_files[spk].append(f)
    speaker_label[spk] = lbl

aff_spk = [s for s,l in speaker_label.items() if l==1]  # 15
ctl_spk = [s for s,l in speaker_label.items() if l==0]  # 13
assert len(aff_spk)==15 and len(ctl_spk)==13, "Unexpected speaker totals"

# ── deterministic stratified split 2/2 val + 2/2 test ──────────────────
random.shuffle(aff_spk); random.shuffle(ctl_spk)
val_spk  = aff_spk[:2] + ctl_spk[:2]
test_spk = aff_spk[2:4] + ctl_spk[2:4]
train_spk = aff_spk[4:] + ctl_spk[4:]

def files_of(spk_list): return [f for s in spk_list for f in speaker_files[s]]
train_files, val_files, test_files = map(files_of, (train_spk,val_spk,test_spk))

# ── leakage audit ───────────────────────────────────────────────────────
def audit():
    sets = {"train":train_files, "val":val_files, "test":test_files}
    # 1) file overlap
    for a in sets:
        for b in sets:
            if a>=b: continue
            if set(sets[a]) & set(sets[b]):
                print(f"❌ File leakage between {a} & {b}"); sys.exit(1)
    # 2) speaker overlap
    spk_sets = {k:{speaker_id(os.path.basename(f)) for f in v} for k,v in sets.items()}
    for a in spk_sets:
        for b in spk_sets:
            if a>=b: continue
            if spk_sets[a] & spk_sets[b]:
                print(f"❌ Speaker leakage between {a} & {b}"); sys.exit(1)
    # 3) val/test 2+2 class balance
    for name, spks in [("val",val_spk), ("test",test_spk)]:
        c = Counter(speaker_label[s] for s in spks)
        if c[1]!=2 or c[0]!=2:
            print(f"❌ {name} split not 2 aff / 2 ctl"); sys.exit(1)
    print("✔ Leakage audit passed.")
audit()

# ── data loaders ────────────────────────────────────────────────────────
def make_loader(files, balance=False):
    ds = MFCCDataset(files)
    if not balance:
        return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
    labels = [1 if f.endswith("_a.npy") else 0 for f in files]
    w = 1/np.bincount(labels); samp_w = [w[l] for l in labels]
    sampler = WeightedRandomSampler(samp_w, len(files), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=True)

train_loader = make_loader(train_files, balance=True)
val_loader   = make_loader(val_files)
test_loader  = make_loader(test_files)

# ── model ───────────────────────────────────────────────────────────────
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc    = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.3)

# ── training loop with early stop ───────────────────────────────────────
best_state, best_val, stale = None, 0, 0
for epoch in range(1, EPOCHS+1):
    for phase, loader, train in [("train",train_loader,True), ("val",val_loader,False)]:
        model.train(train); torch.set_grad_enabled(train)
        losses, y_true, y_pred = 0., [], []
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out,y)
            if train: optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item()*x.size(0)
            y_true += y.cpu().tolist(); y_pred += out.argmax(1).cpu().tolist()
        if not train:
            val_acc = accuracy_score(y_true, y_pred)
            scheduler.step(losses/len(loader.dataset))
            if val_acc > best_val: best_val, best_state, stale = val_acc, model.state_dict().copy(), 0
            else: stale += 1
    print(f"[{epoch:02d}] val acc {best_val:.3f}  stale {stale}")
    if stale >= PATIENCE:
        print("⏹ Early stopping"); break

# ── evaluation on test ─────────────────────────────────────────────────
model.load_state_dict(best_state); model.eval()
probs, preds, labels = [], [], []
with torch.no_grad():
    for x,y in test_loader:
        out = model(x.to(DEVICE))
        probs += F.softmax(out,1)[:,1].cpu().tolist()
        preds += out.argmax(1).cpu().tolist()
        labels+= y.tolist()

print("\\nTEST Accuracy {:.3f}  AUC {:.3f}".format(
    accuracy_score(labels, preds), roc_auc_score(labels, probs)))
print("Confusion matrix:\\n", confusion_matrix(labels, preds))
print("\\nClassification report:\\n", classification_report(labels, preds, digits=3))

torch.save(best_state, os.path.join(OUT_DIR, "resnet18_best.pt"))
with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
    json.dump({"best_val_acc": best_val}, f, indent=2)
print("\\nModel and history saved to", OUT_DIR)
