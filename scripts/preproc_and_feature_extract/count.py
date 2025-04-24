#!/usr/bin/env python3
"""
count_utterances_fixed.py

Count the number of speakers and utterances (MFCC .npy files) in the UASpeech dataset.
Assumes filenames like UAS_f_04_2111_a.npy where:
    - 'UAS_f_04' is the speaker ID
    - Each file corresponds to one utterance

Usage:
    python count_utterances_fixed.py
"""

import os
import glob
import re
import collections

# Hardcoded root directory
ROOT_DIR = '/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/'

def extract_speaker_id(fname):
    """Extracts speaker ID from filename, e.g., UAS_f_04_2111_a.npy -> UAS_f_04"""
    base = os.path.basename(fname)
    parts = base.split('_')
    if len(parts) < 4:
        return None
    return '_'.join(parts[:3])

def main():
    counter = collections.Counter()
    for fp in glob.glob(os.path.join(ROOT_DIR, "**", "*.npy"), recursive=True):
        sid = extract_speaker_id(fp)
        if sid:
            counter[sid] += 1

    print(f"Total speakers: {len(counter)}\n")
    print(f'{"Speaker ID":<15}{"Utterances":>12}')
    print("-" * 27)
    for sid, count in sorted(counter.items()):
        print(f"{sid:<15}{count:>12}")

if __name__ == "__main__":
    main()
