#!/usr/bin/env python3
import pandas as pd

# Load the CSV file
csv_path = "/home/the_fat_cat/Documents/data/features/MFCCs/UASpeech/mfcc_features_39D_UASpeech.csv"
df = pd.read_csv(csv_path)

# Extract the status ('a' or 'c') from the filename
df['status'] = df['filename'].str.extract(r'_([ac])\.wav$')

# Count the number of each status
counts = df['status'].value_counts()

# Print results
print("Afflicted count:", counts.get('a', 0))
print("Control count:", counts.get('c', 0))
