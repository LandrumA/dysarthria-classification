#!/usr/bin/env python3
"""
Fast ResNet-18 classifier for 2-D MFCC .npy files

Key features
────────────
• Per-gender training (no mixing)
• Subject-level 80/20 split (no speaker leakage)
• Word-matching across classes
• Fixed-width padded spectrograms
• **NEW** – subsample a fixed number of clips per speaker to speed up runs
"""

# ─── USER CONFIG ─────────────────────────────────────────────────────────── #
ROOT_DIR               = r"/Users/vrraci/Desktop/Projects/dysarthria/data/features/2d"
EPOCHS                 = 25          # reduce if still too slow
BATCH                  = 32
LR                     = 1e-3
SEED                   = 42
FIXED_WIDTH            = 128         # pad or truncate in time dimension
SUBSAMPLE_PER_SPEAKER  = 100        # clips per speaker (adjust for speed)
# ─────────────────────────────────────────────────────────────────────────── #

import os, re, warnings, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("Agg")

PLOT_DIR = "plots_resnet"
os.makedirs(PLOT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── DATA DISCOVERY ──────────────────────────────────────────────────────── #
def discover_dataset(root: str) -> pd.DataFrame:
    records = []
    root_path = Path(root)
    pat = re.compile(r"^[A-Za-z0-9]+_([A-Za-z0-9]+)_([A-Za-z0-9]+)")  # word_id pattern

    for file in root_path.rglob("*.npy"):
        try:
            group, gender, subject = file.parts[-4:-1]
        except ValueError:
            continue
        m = pat.match(file.stem)
        if not m:
            continue
        word_id = f"{m.group(1)}_{m.group(2)}"
        records.append(dict(
            group=group, gender=gender, subject=subject,
            word_id=word_id, filename=str(file)
        ))

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No .npy files found under {root}")
    return df


def retain_matched_words(df: pd.DataFrame) -> pd.DataFrame:
    freq = df.groupby(['word_id', 'group']).size().unstack(fill_value=0)
    matched = freq[(freq['afflicted'] > 0) & (freq['control'] > 0)].index
    return df[df['word_id'].isin(matched)].copy()


def subject_split(df: pd.DataFrame, seed: int):
    df = df.copy()
    df['label'] = df['group'].map({'control': 0, 'afflicted': 1})
    Xp, y, g = df['filename'].values, df['label'].values, df['subject'].values
    gs = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gs.split(Xp, y, g))
    return Xp[tr_idx], Xp[te_idx], y[tr_idx], y[te_idx]


# ─── TORCH DATASET ───────────────────────────────────────────────────────── #
class MFCC2DDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx])              # [H, W]

        # optional simple augmentation: time-flip 50 %
        if self.augment and np.random.rand() < 0.5:
            arr = np.flip(arr, axis=1).copy()

        # pad / truncate to FIXED_WIDTH
        h, w = arr.shape
        if w < FIXED_WIDTH:
            arr = np.pad(arr, ((0, 0), (0, FIXED_WIDTH - w)), mode='constant')
        else:
            arr = arr[:, :FIXED_WIDTH]

        x = torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]
        y = torch.tensor(self.labels[idx])
        return x, y


# ─── MODEL ───────────────────────────────────────────────────────────────── #
def build_resnet() -> nn.Module:
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Linear(net.fc.in_features, 2)
    return net


# ─── TRAIN / EVAL ────────────────────────────────────────────────────────── #
def train_epoch(net, loader, optim, crit):
    net.train(); running = 0.
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        loss = crit(net(xb), yb)
        loss.backward(); optim.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def evaluate(net, loader, crit):
    net.eval()
    soft = nn.Softmax(dim=1)
    losses, preds, probs, labs = [], [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = net(xb)
        losses.append(crit(out, yb).item() * xb.size(0))
        p = soft(out)[:, 1]
        preds.extend((p >= .5).cpu().numpy())
        probs.extend(p.cpu().numpy())
        labs.extend(yb.cpu().numpy())
    loss = sum(losses) / len(loader.dataset)
    return loss, np.array(preds), np.array(probs), np.array(labs)


def save_plots(tag, labs, preds, probs, auc, tr_loss, vl_loss):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        labs, preds, ax=ax, cmap="Blues",
        display_labels=["Control", "Afflicted"], colorbar=False)
    ax.set_title(f"{tag} – Confusion Matrix")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, f"{tag}_cm.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(labs, probs, ax=ax)
    ax.set_title(f"{tag} – ROC (AUC={auc:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, f"{tag}_roc.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(tr_loss, label="train"); ax.plot(vl_loss, label="val")
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{tag} – Loss Curve")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"{tag}_loss.png"))
    plt.close(fig)


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────── #
def run_pipeline():
    df = discover_dataset(ROOT_DIR)

    for gender in ["male", "female"]:
        df_g = retain_matched_words(df[df.gender == gender])
        if df_g.empty:
            print(f"[!] {gender}: nothing to train on.")
            continue

        # subsample to speed up training
        df_g = (
            df_g.groupby("subject", group_keys=False)
                 .apply(lambda x: x.sample(
                     min(SUBSAMPLE_PER_SPEAKER, len(x)), random_state=SEED))
        )

        tr_p, te_p, y_tr, y_te = subject_split(df_g, SEED)
        dl_tr = DataLoader(MFCC2DDataset(tr_p, y_tr, augment=True),
                           batch_size=BATCH, shuffle=True, num_workers=2)
        dl_te = DataLoader(MFCC2DDataset(te_p, y_te, augment=False),
                           batch_size=BATCH, shuffle=False, num_workers=2)

        net = build_resnet().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        tr_loss, vl_loss = [], []
        print(f"\n▶ {gender.capitalize()} ({len(tr_p)} train / {len(te_p)} val)")

        for _ in tqdm(range(EPOCHS), bar_format=" {l_bar}{bar}| {n_fmt}/{total_fmt}"):
            tr_loss.append(train_epoch(net, dl_tr, opt, crit))
            vl_loss.append(evaluate(net, dl_te, crit)[0])

        _, preds, probs, labs = evaluate(net, dl_te, crit)
        acc  = accuracy_score(labs, preds)
        prec = precision_score(labs, preds, zero_division=0)
        rec  = recall_score(labs, preds, zero_division=0)
        auc  = roc_auc_score(labs, probs)

        print(f"{gender.capitalize():6s} | "
              f"Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | AUC {auc:.3f}")

        save_plots(gender, labs, preds, probs, auc, tr_loss, vl_loss)

    print("\nAll plots saved to:", os.path.abspath(PLOT_DIR))


# ─── LAUNCH ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    run_pipeline()
