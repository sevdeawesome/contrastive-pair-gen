import pandas as pd
import os

# Path to the downloaded dataset
dataset_path = "/home/sevdeawesome/.cache/kagglehub/datasets/rtatman/ironic-corpus/versions/1/irony-labeled.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData Types:")
print(df.dtypes)

print("\nLabel distribution:")
print(df['label'].value_counts() if 'label' in df.columns else "No 'label' column")

print("\nSample ironic examples (if label column exists):")
if 'label' in df.columns:
    ironic = df[df['label'] == 1].head(5)
    print(ironic)

print("\nSample non-ironic examples:")
if 'label' in df.columns:
    non_ironic = df[df['label'] == 0].head(5)
    print(non_ironic)

print("\nColumn value counts for all columns:")
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))
