#!/usr/bin/env python3
"""
Conditional DC-GAN for UASpeech 2-D MFCCs
─────────────────────────────────────────
• Input  shape : (1, 39, 300)  (C, F, T)
• Output shape : (1, 39, 300)

Dataset root : /home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D
Model  output : /home/the_fat_cat/Documents/GitHub/dysarthria-classification/models/UASpeech/GAN
"""

# ── imports ─────────────────────────────────────────────────────────────
import os, glob, re, random, math, itertools, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

# ── hyper-parameters ────────────────────────────────────────────────────
DATA_DIR        = Path("/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D")
OUT_DIR         = Path("/home/the_fat_cat/Documents/GitHub/dysarthria-classification/models/UASpeech/GAN")
LATENT_DIM      = 128
BATCH_SIZE      = 64
EPOCHS          = 200
LR              = 2e-4
BETAS           = (0.5, 0.999)
LABEL_SMOOTH    = 0.9            # real label smoothing
NUM_WORKERS     = 4
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────
def base_speaker_id(fname: str) -> str:
    """
    Unique speaker ID = everything except <recordingID>
    e.g. UAS_m_teen_01_0003_a.npy  →  UAS_m_teen_01_a
    """
    parts = Path(fname).stem.split("_")
    return "_".join(parts[:4] + [parts[-1]])   # drop recordingID (index -2)

def load_npy(fp: Path) -> torch.Tensor:
    arr = np.load(fp)
    # ensure (C,F,T) = (1,39,300)
    if arr.shape != (39, 300):
        raise ValueError(f"Bad shape {arr.shape} for {fp.name}")
    x = torch.from_numpy(arr).float().unsqueeze(0)      # add channel dim
    # per-clip CMVN
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x

# ── Dataset ─────────────────────────────────────────────────────────────
class UASpeechMFCC(Dataset):
    def __init__(self, file_paths):
        self.files = file_paths
        self.labels = [1 if fp.stem.endswith("_a") else 0 for fp in file_paths]
        self.groups = [base_speaker_id(fp.name) for fp in file_paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = load_npy(self.files[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ── Conditional channels ───────────────────────────────────────────────
def add_label_channel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Concatenate a constant-value channel encoding the class label.
    x : (B,1,39,300)  y : (B,)
    returns (B,2,39,300)
    """
    label_chan = y.view(-1, 1, 1, 1).float().expand(-1, 1, 39, 300)
    return torch.cat([x, label_chan], dim=1)

# ── Networks ────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(2, LATENT_DIM)
        self.net = nn.Sequential(
            # input (B, LATENT_DIM, 1, 1)
            nn.ConvTranspose2d(LATENT_DIM, 512,  kernel_size=(5, 4), stride=(1, 1), padding=(0, 0)),    # (512, 5, 4)
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (256,10, 8)
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=(4,4), stride=(2,2), padding=1),  # (128,20,16)
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64,  kernel_size=(4,4), stride=(2,2), padding=1),  # (64,40,32)
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # keep shape
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z * self.label_emb(labels)          # element-wise modulation
        z = z.view(z.size(0), LATENT_DIM, 1, 1)
        out = self.net(z)
        # shape fix: crop height to 39, pad width to 300 if needed
        out = out[:, :, :39, :300]
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),      # (64,19,150)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),        # (128,9,75)
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),       # (256,4,37)
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256,512,kernel_size=4,stride=(2,2),padding=1),   # (512,2,18)
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512*2*18, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        x = add_label_channel(x, labels)
        return self.net(x)

# ── data split & loaders ────────────────────────────────────────────────
all_files = sorted(DATA_DIR.glob("*.npy"))
dataset   = UASpeechMFCC(all_files)

groups = np.array(dataset.groups)
gss     = GroupShuffleSplit(n_splits=1, train_size=0.6, test_size=0.4, random_state=SEED)
train_idx, temp_idx = next(gss.split(np.zeros(len(groups)), groups=groups))

# split temp 0.4 → val/test 0.2/0.2
temp_groups = groups[temp_idx]
gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss2.split(np.zeros(len(temp_groups)), groups=temp_groups))
val_idx  = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

train_loader = DataLoader(Subset(dataset, train_idx),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(Subset(dataset, val_idx),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# ── model, optimizers, loss ─────────────────────────────────────────────
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

criterion = nn.BCELoss()

fixed_noise  = torch.randn(16, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor([0]*8 + [1]*8, device=DEVICE)

# ── training loop ───────────────────────────────────────────────────────
def save_checkpoint(epoch):
    torch.save({"G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict()},
               OUT_DIR / f"epoch_{epoch:03d}.pt")

for epoch in range(1, EPOCHS + 1):
    G.train(); D.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=False)
    for real, labels in pbar:
        real, labels = real.to(DEVICE), labels.to(DEVICE)
        bsz = real.size(0)
        valid = torch.full((bsz,1), LABEL_SMOOTH, device=DEVICE)
        fake  = torch.zeros((bsz,1), device=DEVICE)

        # ── Train D ────────────────────────────────────────────────────
        opt_D.zero_grad()

        pred_real = D(real, labels)
        loss_real = criterion(pred_real, valid)

        z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
        gen_labels = torch.randint(0,2,(bsz,), device=DEVICE)
        gen_imgs   = G(z, gen_labels).detach()

        pred_fake = D(gen_imgs, gen_labels)
        loss_fake = criterion(pred_fake, fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # ── Train G ────────────────────────────────────────────────────
        opt_G.zero_grad()

        z = torch.randn(bsz, LATENT_DIM, device=DEVICE)
        gen_labels = torch.randint(0,2,(bsz,), device=DEVICE)
        gen_imgs   = G(z, gen_labels)

        pred = D(gen_imgs, gen_labels)
        loss_G = criterion(pred, valid)       # trick: want discriminator to think they're real
        loss_G.backward()
        opt_G.step()

        pbar.set_postfix(loss_D=f"{loss_D.item():.4f}", loss_G=f"{loss_G.item():.4f}")

    # ── simple val check (discriminator accuracy) ─────────────────────
    D.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (D(x, y) > 0.5).long()
            correct += (pred.squeeze() == 1).sum().item()   # real labelled as real
            total   += y.size(0)
    acc = correct / total
    print(f"Epoch {epoch:03d} │ D(real) acc on val: {acc:.3f}")

    if epoch % 25 == 0 or epoch == EPOCHS:
        save_checkpoint(epoch)

print("Training complete. Checkpoints saved in", OUT_DIR)
