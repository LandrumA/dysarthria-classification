#!/usr/bin/env python3
"""
Speaker-independent ResNet-18 for UASpeech MFCCs  (final)
──────────────────────────────────────────────────────────
"""

# ── imports ─────────────────────────────────────────────────────────────
import os, glob, random, json, sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet18
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

# ── CONFIG ──────────────────────────────────────────────────────────────
ROOT_2D   = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"
MODEL_DIR = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/models/UASpeech/resNet"
CURVE_DIR = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/results/UASpeech/curves"
LOG_DIR   = "/home/the_fat_cat/Documents/GitHub/dysarthria-classification/results/UASpeech/logs"
EPOCHS, BATCH, LR, MAX_LEN, PATIENCE = 25, 32, 1e-3, 300, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
for d in (MODEL_DIR, CURVE_DIR, LOG_DIR): os.makedirs(d, exist_ok=True)

# ── helper functions ────────────────────────────────────────────────────
def speaker_id(fname: str) -> str:
    base = os.path.basename(fname).replace(".npy", "")
    p1, p2, p3, _, p5 = base.split("_")
    return f"{p1}_{p2}_{p3}_{p5}"        # e.g. UAS_f_04_a

def pad_trunc(x, max_len=MAX_LEN):
    return x[:, :max_len] if x.shape[1] >= max_len else np.pad(x, ((0,0),(0,max_len-x.shape[1])))

class MFCCDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        x = np.load(path);  std = x.std()
        if std < 1e-5:
            raise ValueError(f"Low-variance clip: {path}")
        x = (pad_trunc(x) - x.mean()) / (std + 1e-8)
        y = 1 if path.endswith("_a.npy") else 0
        return torch.from_numpy(x).float().unsqueeze(0), y

# ── gather true speakers ────────────────────────────────────────────────
spk_files, spk_label = defaultdict(list), {}
for f in glob.glob(os.path.join(ROOT_2D, "*.npy")):
    spk = speaker_id(f);  lab = 1 if spk.endswith("_a") else 0
    spk_files[spk].append(f);  spk_label[spk] = lab
aff, ctl = [s for s,l in spk_label.items() if l==1], [s for s,l in spk_label.items() if l==0]

# fixed 2+2 val/test
random.Random(SEED).shuffle(aff); random.Random(SEED).shuffle(ctl)
val_spk  = aff[:2]+ctl[:2]; test_spk = aff[2:4]+ctl[2:4]; train_spk = aff[4:]+ctl[4:]

files_of = lambda S: [f for s in S for f in spk_files[s]]
train_f, val_f, test_f = map(files_of, (train_spk,val_spk,test_spk))

# ── data loaders ────────────────────────────────────────────────────────
def loader(files, balance=False):
    ds = MFCCDataset(files)
    if not balance:
        return DataLoader(ds, BATCH, shuffle=False, num_workers=4, pin_memory=True)
    lbls = [1 if f.endswith("_a.npy") else 0 for f in files]
    w = 1/np.bincount(lbls); samp = [w[l] for l in lbls]
    return DataLoader(ds, BATCH,
                      sampler=WeightedRandomSampler(samp,len(files),replacement=True),
                      num_workers=4, pin_memory=True)

train_loader = loader(train_f, balance=True)
val_loader   = loader(val_f)
test_loader  = loader(test_f)

# ── model, loss, opt ────────────────────────────────────────────────────
model = resnet18(weights=None)
model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
model.fc    = nn.Linear(model.fc.in_features,2)
model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.3)

history = {"train_loss":[],"val_loss":[]}

# ── training loop with early stop ───────────────────────────────────────
best_state, best_val, stale = None, 0, 0
for epoch in range(1, EPOCHS+1):
    # --- train
    model.train(); tloss=0
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb); loss = loss_fn(out,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tloss += loss.item()*xb.size(0)
    history["train_loss"].append(tloss/len(train_loader.dataset))
    # --- val
    model.eval(); vloss, y_true, y_pred = 0, [], []
    with torch.no_grad():
        for xb,yb in val_loader:
            out = model(xb.to(DEVICE)); loss = loss_fn(out,yb.to(DEVICE))
            vloss += loss.item()*xb.size(0)
            y_true+=yb.tolist(); y_pred+=out.argmax(1).cpu().tolist()
    vloss/=len(val_loader.dataset); history["val_loss"].append(vloss)
    val_acc = accuracy_score(y_true,y_pred); sched.step(vloss)
    print(f"[{epoch:02d}] val acc {val_acc:.3f}")
    if val_acc>best_val: best_val,val_epoch,stale=val_acc,epoch,0; best_state=model.state_dict().copy()
    else: stale+=1
    if stale>=PATIENCE: print("⏹ Early stop"); break

# ── test eval ───────────────────────────────────────────────────────────
model.load_state_dict(best_state); model.eval()
probs,preds,labs=[],[],[]
with torch.no_grad():
    for xb,yb in test_loader:
        out=model(xb.to(DEVICE))
        probs+=F.softmax(out,1)[:,1].cpu().tolist()
        preds+=out.argmax(1).cpu().tolist()
        labs +=yb.tolist()

acc=accuracy_score(labs,preds); auc=roc_auc_score(labs,probs)
print(f"TEST acc {acc:.3f}  AUC {auc:.3f}")

# ── save model ----------------------------------------------------------
torch.save(best_state, os.path.join(MODEL_DIR,"resnet18_best.pt"))

# ── plots ---------------------------------------------------------------
plt.figure();                # ROC
fpr,tpr,_=roc_curve(labs,probs); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.savefig(os.path.join(CURVE_DIR,"roc.png"))

plt.figure();                # loss curve
plt.plot(history["train_loss"],label="train"); plt.plot(history["val_loss"],label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss"); plt.legend()
plt.savefig(os.path.join(CURVE_DIR,"loss.png"))

# ── confusion matrix & report ------------------------------------------
cm = confusion_matrix(labs,preds)
plt.figure(); plt.imshow(cm,interpolation="nearest"); plt.title("Confusion"); plt.colorbar()
plt.xticks([0,1],["Control","Afflicted"]); plt.yticks([0,1],["Control","Afflicted"])
plt.xlabel("Pred"); plt.ylabel("True")
plt.savefig(os.path.join(LOG_DIR,"confusion.png"))

with open(os.path.join(LOG_DIR,"classification_report.txt"),"w") as f:
    f.write(classification_report(labs,preds,digits=3))

print("Artifacts saved to:", MODEL_DIR, CURVE_DIR, LOG_DIR)
