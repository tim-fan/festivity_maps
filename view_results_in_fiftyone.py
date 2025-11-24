"""
View foreground segmentation results in FiftyOne.

Loads the inference results CSV and displays images with mean_probability scores.
"""

import fiftyone as fo
import pandas as pd
import os
from pathlib import Path

# Define the path to the CSV file
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "inference_results.csv")
DATASET_NAME = "foreground_segmentation_results"

# Sanity check: ensure the CSV exists
if not os.path.isfile(CSV_PATH):
    print(f"Error: CSV file not found at {CSV_PATH}")
    print(f"Please run the training script with --csv-output flag first")
    exit(1)

# Load the CSV data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from CSV")

# Create a FiftyOne dataset
dataset = fo.Dataset(name=DATASET_NAME, overwrite=True)

# Add samples to the dataset
samples = []
for idx, row in df.iterrows():
    # Get the image path (should already be absolute from the CSV)
    image_path = row['image_path']
    
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        continue
    
    # Create a sample with the image and metadata
    sample = fo.Sample(filepath=image_path)
    sample['mean_probability'] = float(row['mean_probability'])
    
    samples.append(sample)

# Add all samples to the dataset
dataset.add_samples(samples)

# Create index on mean_probability so it appears in the sort dropdown
dataset.create_index("mean_probability")

print(f"Successfully created dataset '{dataset.name}' with {len(dataset)} samples.")
print("You can now sort by 'mean_probability' in the FiftyOne app.")

session = fo.launch_app(dataset)
session.wait()  # Keep the session open until manually closed