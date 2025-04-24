import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== CONFIGURATION ==========
INPUT_DIRS = {
    "NDDS":    "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/NDDS-single-words",
    "TORGO":   "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/TORGO-single-words",
    "UASpeech":"/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/UASpeech-single-words",
}

BASE_OUTPUT_DIR_1D = "/home/the_fat_cat/Documents/data/features/MFCCs"
BASE_OUTPUT_DIR_2D = "/home/the_fat_cat/Documents/data/features/MFCCs"
SR = 16000
N_MFCC = 13
DELTA_WIDTH = 9  # default window width for delta feature

print("Extracting MFCC + Delta + Delta-Delta features...")

total_files = sum(len(files) for folder in INPUT_DIRS.values() for _, _, files in os.walk(folder))
pbar = tqdm(total=total_files, desc="Overall Progress", unit="file")

for dataset, folder in INPUT_DIRS.items():
    output_dir_1d = os.path.join(BASE_OUTPUT_DIR_1D, dataset)
    output_dir_2d = os.path.join(BASE_OUTPUT_DIR_2D, dataset)
    os.makedirs(output_dir_1d, exist_ok=True)
    os.makedirs(output_dir_2d, exist_ok=True)

    rows = []  # store 1D feature rows

    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)
            try:
                y, _ = librosa.load(file_path, sr=SR)
                mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
                n_frames = mfcc.shape[1]

                # Skip files with too few frames for delta computation
                if n_frames < DELTA_WIDTH:
                    print(f"Skipping {file_path}: only {n_frames} frames (<{DELTA_WIDTH})")
                    pbar.update(1)
                    continue

                # Compute delta and delta-delta
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc_combined = np.vstack([mfcc, delta, delta2])  # Shape: (39, T)

                # ----- 1D Feature: average across time -----
                mfcc_1d = np.mean(mfcc_combined, axis=1)
                rows.append([file] + mfcc_1d.tolist())

                # ----- 2D Feature: save per-clip -----
                out_path_2d = os.path.join(output_dir_2d, file.replace(".wav", ".npy"))
                np.save(out_path_2d, mfcc_combined)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
            finally:
                pbar.update(1)

    # Save CSV for this dataset
    columns = ["filename"] + [f"mfcc{i+1}" for i in range(13)] + [f"delta{i+1}" for i in range(13)] + [f"delta2_{i+1}" for i in range(13)]
    df = pd.DataFrame(rows, columns=columns)
    csv_path = os.path.join(output_dir_1d, f"mfcc_features_39D_{dataset}.csv")
    df.to_csv(csv_path, index=False)

pbar.close()
print("Feature extraction complete.")
