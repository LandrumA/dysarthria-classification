#!/usr/bin/env python3
import re
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
INPUT_BASE       = Path("/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_original/NDDS/NDDS/SPEECH/SENT")
OUTPUT_BASE      = Path("/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/NDDS-single-words")
MAX_DURATION_SEC = 6.0       # skip segments longer than this (in seconds)
SILENCE_DB       = -40.0     # skip segments quieter than this (RMS in dB)
# ────────────────────────────────────────────────────────────────────────────────

def is_valid_segment(y_segment: np.ndarray, sr: int) -> bool:
    duration = y_segment.shape[0] / sr
    if duration > MAX_DURATION_SEC:
        return False
    energy_db = librosa.amplitude_to_db(librosa.feature.rms(y=y_segment), ref=np.max).mean()
    return energy_db >= SILENCE_DB

def process_speaker_folder(speaker_dir: Path, is_control: bool):
    speaker_id = speaker_dir.name.upper()
    affliction = "c" if is_control else "a"
    wav_dir = speaker_dir / "WAV"
    seg_dir = speaker_dir / "SEG"
    txt_dir = speaker_dir / "TXT"

    if not (wav_dir.is_dir() and seg_dir.is_dir() and txt_dir.is_dir()):
        print(f"[SKIP] {speaker_id}: missing WAV/SEG/TXT")
        return

    wav_files = sorted([f for f in wav_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])
    if not wav_files:
        print(f"[SKIP] {speaker_id}: no .wav files found")
        return

    print(f"[START] {speaker_id} ({'control' if is_control else 'afflicted'})")
    counter = 1
    extracted = 0

    for wav_path in tqdm(wav_files, desc=speaker_id, unit="file"):
        base = wav_path.stem
        seg_path = seg_dir / f"{base}.SEG"
        txt_path = txt_dir / f"{base}.TXT"
        if not (seg_path.exists() and txt_path.exists()):
            print(f"  [MISSING] {base}: .SEG or .TXT not found")
            continue

        # load full sentence
        y, sr = librosa.load(str(wav_path), sr=None)
        words = txt_path.read_text().strip().split()

        for line in seg_path.read_text().splitlines():
            parts = re.split(r"\s+", line.strip())
            if not parts or not parts[0].startswith("W"):
                continue

            idx = int(parts[0][1:]) - 1
            if idx < 0 or idx >= len(words):
                continue

            # convert microseconds to sample indices
            start_us = float(parts[1])
            end_us   = float(parts[2])
            start_sample = int(start_us / 1e6 * sr)
            end_sample   = int(end_us   / 1e6 * sr)

            if end_sample <= start_sample or end_sample > len(y):
                continue

            segment = y[start_sample:end_sample]
            if not is_valid_segment(segment, sr):
                continue

            out_name = f"NDDS_m_{speaker_id}_{counter:04d}_{affliction}.wav"
            sf.write(str(OUTPUT_BASE / out_name), segment, sr)
            counter += 1
            extracted += 1

    print(f"[DONE] {speaker_id}: extracted {extracted} segments\n")

def main():
    print(">>> SCRIPT STARTED <<<")
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    control_done = False

    for folder in sorted(INPUT_BASE.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.upper()
        if name.startswith("JP"):
            if control_done:
                continue
            process_speaker_folder(folder, is_control=True)
            control_done = True
        else:
            process_speaker_folder(folder, is_control=False)

if __name__ == "__main__":
    main()
