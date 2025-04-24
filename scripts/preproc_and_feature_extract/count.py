from collections import defaultdict, Counter
import os, glob

ROOT_2D = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/2D"

def extract_speaker_id(fname):
    parts = fname.replace(".npy", "").split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected filename format: {fname}")
    return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[4]}"  # e.g. UAS_f_04_a

speaker_label = {}

for f in glob.glob(os.path.join(ROOT_2D, "*.npy")):
    fname = os.path.basename(f)
    speaker_id = extract_speaker_id(fname)
    label = 1 if speaker_id.endswith("_a") else 0
    speaker_label[speaker_id] = label  # deduplicates by speaker

# Count unique speakers
counts = Counter(speaker_label.values())
print("✔ True unique speaker counts:")
print(f"Afflicted: {counts[1]}")
print(f"Control  : {counts[0]}")
print(f"Total    : {len(speaker_label)} speakers")
