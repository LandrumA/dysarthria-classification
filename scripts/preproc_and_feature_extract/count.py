#!/usr/bin/env python3
import pandas as pd

# Load the CSV file
csv_path = "/home/the_fat_cat/Documents/data/features/MFCCs/TORGO/mfcc_features_39D_TORGO.csv"
df = pd.read_csv(csv_path)

# Extract the status ('a' or 'c') from the filename
df['status'] = df['filename'].str.extract(r'_([ac])\.wav$')

# Extract speaker ID (e.g., F02, M01, etc.)
df['speaker'] = df['filename'].str.extract(r'_(F|M|FC|MC)?(\d{2})_')[1]
df['group']   = df['filename'].str.extract(r'_(F|M|FC|MC)?(\d{2})_')[0]
df['speaker_id'] = df['group'].fillna('') + df['speaker']

# Count total recordings
counts = df['status'].value_counts()

# Count unique speakers per class
speaker_counts = df.groupby('status')['speaker_id'].nunique()

# Print results
print("Afflicted count:", counts.get('a', 0))
print("Control count:", counts.get('c', 0))
print("Unique afflicted speakers:", speaker_counts.get('a', 0))
print("Unique control speakers:", speaker_counts.get('c', 0))
