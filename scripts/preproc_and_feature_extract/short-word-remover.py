import os
import soundfile as sf
from pathlib import Path

# CONFIG
TARGET_DIR = Path("/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/NDDS-single-words")
MIN_DURATION_SEC = 0.1  # Adjust as needed

def get_duration(path):
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception as e:
        print(f"  [ERROR] {path.name}: {e}")
        return 0

def main():
    print(f"Scanning {TARGET_DIR} for short clips...")
    removed = 0

    for wav_file in sorted(TARGET_DIR.glob("*.wav")):
        duration = get_duration(wav_file)
        if duration < MIN_DURATION_SEC:
            print(f"  [DELETE] {wav_file.name} ({duration:.3f} sec)")
            wav_file.unlink()
            removed += 1

    print(f"\nDone. Removed {removed} file(s).")

if __name__ == "__main__":
    main()
