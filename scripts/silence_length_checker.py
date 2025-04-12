import os
import librosa
import soundfile as sf

# Set your input directory
INPUT_DIR = '/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/UASpeech-single-words/'

# Threshold for silence (RMS energy)
SILENCE_THRESHOLD = 1e-4  # adjust if needed
MAX_DURATION = 3.0  # seconds

def is_silent(audio, threshold):
    return librosa.feature.rms(y=audio).mean() < threshold

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            try:
                audio, sr = librosa.load(filepath, sr=None)
                duration = len(audio) / sr

                if duration > MAX_DURATION:
                    print(f"Deleting {filename} (too long: {duration:.2f}s)")
                    os.remove(filepath)
                elif is_silent(audio, SILENCE_THRESHOLD):
                    print(f"Deleting {filename} (silent)")
                    os.remove(filepath)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_directory(INPUT_DIR)
