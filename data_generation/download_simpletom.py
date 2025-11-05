"""
Download and convert SimpleToM dataset to JSON format for evaluation.
"""

from datasets import load_dataset
import json
import os

# Load the dataset
print("Loading SimpleToM dataset...")
ds = load_dataset("allenai/SimpleToM", "behavior-qa")

# Explore structure
print("\nDataset structure:")
print(f"Splits: {ds.keys()}")

# Use the first available split
split = list(ds.keys())[0]
print(f"\nFirst example from {split} split:")
print(ds[split][0])
print(f"\nColumn names: {ds[split].column_names}")
print(f"\nNumber of examples in {split}: {len(ds[split])}")

# Print a few more examples to understand the structure
print(f"\nFirst 3 examples:")
for i in range(min(3, len(ds[split]))):
    print(f"\n--- Example {i} ---")
    for key, value in ds[split][i].items():
        print(f"{key}: {value}")
