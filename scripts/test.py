import os
import glob
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Configuration
ROOT_2D = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Extract correct speaker ID (removes 4-digit recording number)
def get_speaker_id(fname):
    parts = fname.replace(".npy", "").split("_")
    return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[4]}"

# Load all files and map to speakers
speaker_files = {}
for f in glob.glob(os.path.join(ROOT_2D, "*.npy")):
    spk = get_speaker_id(os.path.basename(f))
    if spk not in speaker_files:
        speaker_files[spk] = []
    speaker_files[spk].append(f)

# Split speakers
aff = [s for s in speaker_files if s.endswith("_a")]
ctl = [s for s in speaker_files if s.endswith("_c")]
random.shuffle(aff)
random.shuffle(ctl)

val_spk  = aff[:2] + ctl[:2]
test_spk = aff[2:4] + ctl[2:4]

def balanced_sample(spk_list, K=None):
    samples = []
    if K is None:
        K = min(len(speaker_files[s]) for s in spk_list)
    for s in spk_list:
        samples.extend(random.sample(speaker_files[s], K))
    return samples

test_files = balanced_sample(test_spk)

# Load features and true labels
X = []
y_true = []
for f in test_files:
    x = np.load(f)
    if x.std() < 1e-5:
        continue
    x = (x - x.mean()) / (x.std() + 1e-8)
    x = x[:, :300] if x.shape[1] >= 300 else np.pad(x, ((0, 0), (0, 300 - x.shape[1])))
    y = 1 if f.endswith("_a.npy") else 0
    X.append(x)
    y_true.append(y)

# Label shuffle test
y_shuffled = y_true.copy()
random.shuffle(y_shuffled)

# Simulate model prediction = shuffled labels
auc = roc_auc_score(y_true, y_shuffled)
acc = accuracy_score(y_true, [round(p) for p in y_shuffled])

print("Label shuffle test:")
print(f"→ ROC AUC: {auc:.3f}")
print(f"→ Accuracy: {acc:.3f}")
