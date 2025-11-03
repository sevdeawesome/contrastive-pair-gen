#!/usr/bin/env python3
"""
Generate harmfulness contrast pairs using the LessWrong/refusal method.

This pairs harmful prompts from AdvBench with harmless instructions from Alpaca,
matching the approach from: https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction

Data sources:
- Harmless: tatsu-lab/alpaca dataset (HuggingFace)
- Harmful: llm-attacks AdvBench harmful_behaviors.csv (GitHub)
"""

import json
import pandas as pd
import requests
from pathlib import Path

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Setup paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "contrast_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_harmful_behaviors():
    """Download AdvBench harmful behaviors CSV from GitHub."""
    print("Downloading harmful behaviors from AdvBench...")
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    response.raise_for_status()

    # Save to data_generation directory (not the main data/ folder)
    csv_path = Path(__file__).parent / "harmful_behaviors.csv"
    with open(csv_path, 'wb') as f:
        f.write(response.content)

    print(f"Saved to: {csv_path}")
    return pd.read_csv(csv_path)

def load_alpaca_dataset():
    """Load Alpaca dataset from HuggingFace."""
    print("Loading Alpaca dataset from HuggingFace...")

    # Download the JSON version instead of parquet (easier to handle)
    # The alpaca dataset is also available in simpler formats
    print("Fetching Alpaca dataset (this may take a moment)...")
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    print(f"Loaded {len(df)} harmless instructions from Alpaca")
    return df

def generate_contrast_pairs(harmful_df, harmless_df, num_pairs=200):
    """
    Generate contrast pairs by pairing harmful and harmless prompts.

    Note: Unlike semantic pairing, this simply pairs by index - the first harmful
    prompt with the first harmless instruction, etc. This is the LessWrong method.
    """
    print(f"\nGenerating {num_pairs} contrast pairs...")

    pairs = []
    for i in range(num_pairs):
        pair = {
            "harmful": harmful_df.iloc[i]['goal'],
            "harmless": harmless_df.iloc[i]['instruction'],
            # Include category if available
            "harmful_category": harmful_df.iloc[i].get('category', 'unknown')
        }
        pairs.append(pair)

    return pairs

def main():
    print("=" * 60)
    print("Generating LessWrong-style harmfulness contrast pairs")
    print("=" * 60)

    # Load datasets
    harmful_df = download_harmful_behaviors()
    print(f"Loaded {len(harmful_df)} harmful behaviors")

    harmless_df = load_alpaca_dataset()

    # Generate pairs
    pairs = generate_contrast_pairs(harmful_df, harmless_df, num_pairs=200)

    # Save to JSON
    output_file = OUTPUT_DIR / "harmfulness_lesswrong.json"
    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Generated {len(pairs)} contrast pairs")
    print(f"✓ Saved to: {output_file}")

    # Print sample
    print("\n" + "=" * 60)
    print("Sample pairs:")
    print("=" * 60)
    for i, pair in enumerate(pairs[:3]):
        print(f"\nPair {i+1}:")
        print(f"  Harmful:  {pair['harmful'][:80]}...")
        print(f"  Harmless: {pair['harmless'][:80]}...")
        print(f"  Category: {pair['harmful_category']}")

if __name__ == "__main__":
    main()
