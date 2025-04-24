#!/usr/bin/env python3
"""
Extract 39-D MFCC features (13 MFCC + Δ + ΔΔ) from single-word clips,
**skipping any recording shorter than 0.20 s**.

Output layout
─────────────
features/MFCCs/<DATASET>/
    ├── 2D/         # per-clip  (39 × T)  → .npy
    └── 1D/         # per-clip  time-mean → mfcc_39d_<DATASET>.csv

Compatible with downstream training scripts that expect filenames like:
    UAS_m_13_5348_c.npy        (dataset prefix kept)
"""

import os, sys, traceback
import librosa, numpy as np, pandas as pd
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────────────────
DATA_ROOT = "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words"
DATASETS  = ["NDDS", "TORGO", "UASpeech"]          # sub-folder names
OUT_ROOT  = "/home/the_fat_cat/Documents/data/features/MFCCs"

MIN_DUR   = 0.20   # s  → skip recordings shorter than this
SR        = 16_000
N_MFCC    = 13
N_FFT     = 400    # 25 ms window
HOP       = 160    # 10 ms hop  → good time resolution
DELTA_W   = 9      # librosa default (odd)

# ── helpers ─────────────────────────────────────────────────────────────
def ensure(path):
    os.makedirs(path, exist_ok=True)

def extract_39d(y):
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP)
    delta  = librosa.feature.delta(mfcc,  width=DELTA_W, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=DELTA_W, mode="nearest")
    return np.vstack([mfcc, delta, delta2])        # (39, T)

# ── count total wav files for progress bar ──────────────────────────────
total = sum(
    len([f for _, _, fs in os.walk(os.path.join(DATA_ROOT, f"{d}-single-words"))
         for f in fs if f.endswith(".wav")])
    for d in DATASETS
)
pbar = tqdm(total=total, unit="file", desc="Overall")

# ── iterate datasets ────────────────────────────────────────────────────
for ds in DATASETS:
    in_dir  = os.path.join(DATA_ROOT, f"{ds}-single-words")
    out_2d  = os.path.join(OUT_ROOT, ds, "2D")
    out_1d  = os.path.join(OUT_ROOT, ds, "1D")
    ensure(out_2d);  ensure(out_1d)

    rows = []
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if not fname.endswith(".wav"):
                pbar.update();  continue

            fpath = os.path.join(root, fname)
            try:
                y, _ = librosa.load(fpath, sr=SR)
                if len(y) < MIN_DUR * SR:
                    tqdm.write(f"⏭  <0.20 s  {fname}")
                    pbar.update();  continue

                feat = extract_39d(y)

                base = os.path.splitext(fname)[0]
                np.save(os.path.join(out_2d, base + ".npy"), feat)

                rows.append([fname] + feat.mean(axis=1).tolist())

            except Exception as e:
                tqdm.write(f"⚠️  {fname}: {e}")
                tqdm.write(traceback.format_exc())

            finally:
                pbar.update()

    # write CSV (1-D means)
    cols = (["filename"] +
            [f"mfcc{i+1}"   for i in range(13)] +
            [f"delta{i+1}"  for i in range(13)] +
            [f"delta2_{i+1}" for i in range(13)])
    pd.DataFrame(rows, columns=cols)\
      .to_csv(os.path.join(out_1d, f"mfcc_39d_{ds}.csv"), index=False)

pbar.close()
print("Feature extraction finished (>=0.20 s only).")
