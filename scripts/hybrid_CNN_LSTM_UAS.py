#!/usr/bin/env python3
"""
CNN + BiLSTM dysarthria classifier for 2-D MFCCs (UASpeech)

Key guarantees
──────────────
✓ Speaker-independent train/val/test splits (no overlap)  
✓ Per-sample CMVN (mean-0 / std-1)  
✓ Balanced training batches via WeightedRandomSampler  
✓ CNN backbone → flattened over time → BiLSTM → logits  
✓ Saves best model, training history, test metrics & confusion matrix
"""

# ── imports ─────────────────────────────────────────────────────────────
import argparse, json, math, os, random, shutil, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── CONFIG (edit as needed or override via CLI) ─────────────────────────
SEED                 = 42
TEST_SPLIT           = 0.15   # speaker-level ratios
VAL_SPLIT            = 0.15
BATCH_SIZE           = 64
EPOCHS               = 30
PATIENCE             = 5      # early-stopping patience
LR                   = 1e-3
HIDDEN_SIZE          = 256    # per LSTM direction
NUM_LSTM_LAYERS      = 2
NUM_WORKERS          = 4
OUTPUT_DIR           = "outputs_cnn_bilstm"
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
EPS                  = 1e-8

# ── reproducibility ────────────────────────────────────────────────────
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── filename helpers ───────────────────────────────────────────────────
def parse_filename(path: str):
    """Return speaker_id and label (0=control, 1=afflicted) from a UAS filename"""
    stem = Path(path).stem  # e.g. UAS_m_13_1728_a
    parts = stem.split("_")
    speaker_id = "_".join(parts[:3] + [parts[4][0]])  # UAS_m_13_a
    label = 1 if parts[4][0] == "a" else 0
    return speaker_id, label


# ── Dataset ────────────────────────────────────────────────────────────
class MFCCDataset(Dataset):
    def __init__(self, df, cmvn=True):
        self.paths = df["filename"].tolist()
        self.labels = df["label"].tolist()
        self.cmvn = cmvn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx]).astype(np.float32)          # [39, 300]
        if self.cmvn:
            mu = arr.mean()
            sigma = arr.std() + EPS
            arr = (arr - mu) / sigma
        tensor = torch.from_numpy(arr).unsqueeze(0)               # [1, 39, 300]
        return tensor, self.labels[idx]


# ── Model ──────────────────────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
    def __init__(self, hidden=HIDDEN_SIZE, lstm_layers=NUM_LSTM_LAYERS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [B,32,39,300]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=(1,2), padding=1), # time /2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=(1,2), padding=1),# time /2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # after 2 strided steps, time length 300 → 75
        self.time_steps = 300 // 4                               # 75
        self.feature_dim = 128 * 39                              # C*H

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden*2, 2)

    def forward(self, x):
        # x: [B,1,39,300]
        x = self.cnn(x)                                          # [B,128,39,75]
        x = x.permute(0,3,1,2).contiguous()                      # [B,75,128,39]
        x = x.view(x.size(0), x.size(1), -1)                     # [B,75,128*39]
        lstm_out, _ = self.lstm(x)                               # [B,75,512]
        out = lstm_out[:,-1,:]                                   # last time step
        return self.fc(out)                                      # [B,2]


# ── utils ──────────────────────────────────────────────────────────────
def build_dataloaders(csv_path):
    df = pd.read_csv(csv_path)
    df["speaker_id"], df["label"] = zip(*df["filename"].map(parse_filename))

    # unique speakers with label
    spk_df = df[["speaker_id", "label"]].drop_duplicates()

    # stratified speaker split → temp train+val vs test
    spk_trainval, spk_test = train_test_split(
        spk_df,
        test_size=TEST_SPLIT,
        stratify=spk_df["label"],
        random_state=SEED,
    )
    # further split train vs val
    val_ratio = VAL_SPLIT / (1.0 - TEST_SPLIT)
    spk_train, spk_val = train_test_split(
        spk_trainval,
        test_size=val_ratio,
        stratify=spk_trainval["label"],
        random_state=SEED,
    )

    # helper to select rows by speaker set
    def subset(spk_subset):
        return df[df["speaker_id"].isin(spk_subset["speaker_id"])].reset_index(drop=True)

    train_df = subset(spk_train)
    val_df   = subset(spk_val)
    test_df  = subset(spk_test)

    # ─ Weighted sampler for training ─
    class_counts = Counter(train_df["label"].tolist())
    weights = train_df["label"].map(lambda y: 1.0 / class_counts[y]).tolist()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        MFCCDataset(train_df), batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        MFCCDataset(val_df), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        MFCCDataset(test_df), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        n += y.size(0)
    return total_loss/n, total_correct/n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == y).sum().item()
        n += y.size(0)
    return total_loss/n, total_correct/n


def save_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Control","Afflicted"],
                yticklabels=["Control","Afflicted"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_path); plt.close()


# ── main ───────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = build_dataloaders(args.csv)

    model = CNN_BiLSTM().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_loss = math.inf
    epochs_no_improve = 0

    print(f"Training on {len(train_df)} clips "
          f"({train_df['speaker_id'].nunique()} speakers) | "
          f"Val {len(val_df)} | Test {len(test_df)}")

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"]  .append(vl_loss)
        history["train_acc"] .append(tr_acc)
        history["val_acc"]   .append(vl_acc)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"Val loss {vl_loss:.4f} acc {vl_acc:.3f}")

        # early stopping
        if vl_loss + 1e-4 < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), Path(OUTPUT_DIR)/"best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    # save history
    with open(Path(OUTPUT_DIR)/"history.json","w") as f:
        json.dump(history, f, indent=2)

    # ── Test evaluation ────────────────────────────────────────────────
    model.load_state_dict(torch.load(Path(OUTPUT_DIR)/"best_model.pt", map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            preds = logits.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    report = classification_report(y_true, y_pred, target_names=["Control","Afflicted"], digits=4)
    print("\n=== Test classification report ===\n", report)

    # save confusion matrix
    cm_path = Path(OUTPUT_DIR)/"confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Confusion matrix saved → {cm_path.relative_to(Path.cwd())}")

    # also save report to txt
    with open(Path(OUTPUT_DIR)/"test_report.txt","w") as f:
        f.write(report)

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to master CSV with 'filename' column")
    args = parser.parse_args()
    main(args)
