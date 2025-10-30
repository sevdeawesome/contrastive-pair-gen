import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("rtatman/ironic-corpus")

print("Path to dataset files:", path)

# List all files in the downloaded directory
print("\nFiles in dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        print(f"  - {filepath}")
        print(f"    Size: {os.path.getsize(filepath)} bytes")
