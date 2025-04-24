#!/usr/bin/env python3
"""
Preprocess single-word audio clips for dysarthria classification
with a 0.41 s minimum duration cutoff, MMSE-LSA noise reduction,
LUFS normalization, 3 s padding/truncate, and a tqdm progress bar.
"""

import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
import pyloudnorm as pyln
import librosa
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_DIRS = {
    "NDDS":    "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/NDDS-single-words",
    "TORGO":   "/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/TORGO-single-words",
    "UASpeech":"/home/the_fat_cat/Documents/data/dysarthria_raw/dysarthria_raw_single-words/UASpeech-single-words",
}
OUTPUT_DIRS = {
    "NDDS":    "/home/the_fat_cat/Documents/data/preprocessed/NDDS_preprocessed",
    "TORGO":   "/home/the_fat_cat/Documents/data/preprocessed/TORGO_preprocessed",
    "UASpeech":"/home/the_fat_cat/Documents/data/preprocessed/UASpeech_preprocessed",
}

TARGET_SR        = 16000      # Hz
TARGET_DUR       = 3.0        # seconds
MIN_DURATION_SEC = 0.41       # seconds
TARGET_LOUDNESS  = -23.0      # LUFS

def mmse_lsa_denoise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Simplified MMSE-LSA denoiser:
      - STFT
      - estimate noise PSD from first 0.5 s
      - apriori/posterior SNR & Wiener-like gain
      - ISTFT
    """
    # STFT parameters
    n_fft      = 512
    hop_length = 128
    win_length = n_fft

    # Compute complex spectrogram
    D     = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag   = np.abs(D)
    phase = np.angle(D)

    # Noise power spectral density (from first 0.5 s)
    num_noise_frames = int(0.5 * sr / hop_length)
    noise_mag        = mag[:, :num_noise_frames]
    noise_psd        = np.mean(noise_mag**2, axis=1, keepdims=True)

    # Posterior SNR
    epsilon  = 1e-8
    post_snr = mag**2 / (noise_psd + epsilon)

    # A priori SNR (decision-directed approx)
    apr_snr = np.maximum(post_snr - 1.0, 0.0)

    # Gain function (Wiener-like)
    gain = apr_snr / (apr_snr + 1.0 + epsilon)

    # Apply gain & invert
    D_enh = gain * D
    y_enh = librosa.istft(D_enh, hop_length=hop_length, win_length=win_length, length=len(y))

    return y_enh

def preprocess_file(in_path: str, out_path: str):
    # 1. Load & resample
    y, sr_orig = sf.read(in_path)
    if sr_orig != TARGET_SR:
        y = resample(y, int(len(y) * TARGET_SR / sr_orig))
    sr = TARGET_SR

    # 2. Skip too-short
    if len(y) < MIN_DURATION_SEC * sr:
        return

    # 3. Denoise
    y_denoised = mmse_lsa_denoise(y, sr)

    # 4. Loudness normalize
    meter        = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(y_denoised)
    y_loud       = pyln.normalize.loudness(y_denoised, current_lufs, TARGET_LOUDNESS)

    # 5. Pad or truncate to TARGET_DUR
    desired_len = int(TARGET_DUR * sr)
    if len(y_loud) < desired_len:
        y_final = np.pad(y_loud, (0, desired_len - len(y_loud)))
    else:
        y_final = y_loud[:desired_len]

    # 6. Write 16-bit PCM WAV
    sf.write(out_path, y_final, sr, subtype="PCM_16")

def main():
    for key, in_root in INPUT_DIRS.items():
        out_root = OUTPUT_DIRS[key]
        for root, _, files in os.walk(in_root):
            rel_dir = os.path.relpath(root, in_root)
            target_dir = os.path.join(out_root, rel_dir)
            os.makedirs(target_dir, exist_ok=True)

            wavs = [f for f in files if f.lower().endswith(".wav")]
            for fname in tqdm(wavs, desc=f"{key} · {rel_dir}", unit="file"):
                src = os.path.join(root, fname)
                dst = os.path.join(target_dir, fname)
                try:
                    preprocess_file(src, dst)
                except Exception as e:
                    tqdm.write(f"Error processing {src}: {e}")

if __name__ == "__main__":
    main()
