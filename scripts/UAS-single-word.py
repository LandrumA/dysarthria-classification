#!/usr/bin/env python3

#===========================================================================
# UASpeech script for single word extraction (Counter restarts per speaker)
#===========================================================================

import os
import shutil
import re
import librosa
import numpy as np
from datetime import datetime

# === PATHS (Update as needed) ===
INPUT_BASE_AFFLICTED = "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_original/UASpeech/UASpeech_original_FM/UASpeech/audio/original/"
INPUT_BASE_CONTROL   = "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_original/UASpeech/UASpeech_original_C/UASpeech/audio/original/"
OUTPUT_BASE          = "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/UASpeech-single-words/"
LOG_FILE             = os.path.join(OUTPUT_BASE, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

os.makedirs(OUTPUT_BASE, exist_ok=True)

# === UTILITIES ===

def parse_speaker_folder_name(folder_name: str):
    folder_name = folder_name.upper()
    if m := re.match(r"^CF(\d{2})$", folder_name): return ("f", m.group(1), "c")
    if m := re.match(r"^CM(\d{2})$", folder_name): return ("m", m.group(1), "c")
    if m := re.match(r"^F(\d{2})$", folder_name):  return ("f", m.group(1), "a")
    if m := re.match(r"^M(\d{2})$", folder_name):  return ("m", m.group(1), "a")
    return None

def is_valid_audio(file_path, max_duration_sec=6.0, silence_threshold_db=-40.0):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if librosa.get_duration(y=y, sr=sr) > max_duration_sec:
            return False
        energy_db = librosa.amplitude_to_db(librosa.feature.rms(y=y), ref=np.max)
        return energy_db.mean() >= silence_threshold_db
    except Exception as e:
        log(f"⚠️ Error reading {file_path}: {e}")
        return False

def log(message: str):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

# === MAIN PROCESSING ===

def process_speaker_folder(speaker_folder):
    folder_name = os.path.basename(speaker_folder)
    parsed = parse_speaker_folder_name(folder_name)
    if not parsed:
        log(f"Skipping '{folder_name}' — unrecognized folder pattern.")
        return

    gender_letter, speaker_num_str, affliction_letter = parsed
    wav_files = sorted(f for f in os.listdir(speaker_folder) if f.lower().endswith(".wav"))

    # Restart counter at 1 for each speaker
    counter = 1

    for filename in wav_files:
        input_path = os.path.join(speaker_folder, filename)
        if is_valid_audio(input_path):
            idx_formatted = f"{counter:04d}"  # e.g., 0001, 0002, ...
            new_filename = f"UAS_{gender_letter}_{speaker_num_str}_{idx_formatted}_{affliction_letter}.wav"
            output_path = os.path.join(OUTPUT_BASE, new_filename)

            if os.path.exists(output_path):
                log(f"⚠️ File exists, skipping: {output_path}")
                continue

            shutil.copy2(input_path, output_path)
            log(f"✔ Copied: {input_path} -> {output_path}")
            counter += 1
        else:
            log(f"✘ Skipped (silent or too long): {input_path}")

def main():
    for entry in os.scandir(INPUT_BASE_AFFLICTED):
        if entry.is_dir():
            process_speaker_folder(entry.path)
    for entry in os.scandir(INPUT_BASE_CONTROL):
        if entry.is_dir():
            process_speaker_folder(entry.path)

if __name__ == "__main__":
    main()
