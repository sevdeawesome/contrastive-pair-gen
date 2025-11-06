#!/usr/bin/env python3
"""Train a steering vector from contrastive pairs.

Usage:
    python train_steering_vector.py google/gemma-2-7b irony all
    python train_steering_vector.py google/gemma-2-7b theory_of_mind 10,20,30
    python train_steering_vector.py google/gemma-3-4b-it harmfulness 15,18,21
"""

import json
import pickle
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector
import fire


CONTRAST_CONFIGS = {
    'theory_of_mind': {
        'file': 'theory_of_mind.json',
        'extract': lambda item: (
            item['scenario'] + " " + item['high_tom'],
            item['scenario'] + " " + item['low_tom']
        )
    },
    'irony': {
        'file': 'irony.json',
        'extract': lambda item: (item['literal'], item['ironic'])
    },
    'self_other': {
        'file': 'self_other.json',
        'extract': lambda item: (item['self_subject'], item['other_subject'])
    },
    'harmfulness': {
        'file': 'harmfulness.json',
        'extract': lambda item: (
            item['instruction'] + " " + item['harmless'],
            item['instruction'] + " " + item['harmful']
        )
    }
}


def main(model_name, contrast_pair_dataset, layers='all', batch_size=2):
    """Train a steering vector.

    Args:
        model_name: HuggingFace model name (e.g., 'google/gemma-2-7b')
        contrast_pair_dataset: Dataset name (e.g., 'irony', 'theory_of_mind')
        layers: Either 'all' or comma-separated layer indices (e.g., '10,20,30')
        batch_size: Batch size for training (default: 2)
    """
    # Remove .json extension if provided
    dataset_name = contrast_pair_dataset.replace('.json', '')

    if dataset_name not in CONTRAST_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(CONTRAST_CONFIGS.keys())}")

    config = CONTRAST_CONFIGS[dataset_name]

    # Load contrast pairs
    data_path = Path(__file__).parent.parent / 'data' / 'contrast_pairs' / config['file']
    print(f"Loading data from: {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    pairs = [config['extract'](item) for item in data]
    print(f"Loaded {len(pairs)} contrast pairs")

    # Parse layers
    if layers.lower() == 'all':
        layer_list = None
        layer_str = 'all'
    else:
        layer_list = [int(x.strip()) for x in layers.split(',')]
        layer_str = '-'.join(map(str, layer_list))

    print(f"Layers: {layer_str}")

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded")

    # Train steering vector
    print(f"Training steering vector on {len(pairs)} pairs...")
    steering_vec = train_steering_vector(
        model,
        tokenizer,
        pairs,
        layers=layer_list,
        batch_size=batch_size,
        show_progress=True
    )
    print("Training complete")

    # Save to data/trained_vectors/{model_short_name}/{dataset}_{layers}.pkl
    model_short = model_name.split('/')[-1]
    output_dir = Path(__file__).parent.parent / 'data' / 'trained_vectors' / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_{layer_str}.pkl"

    with open(output_path, 'wb') as f:
        pickle.dump(steering_vec, f)

    print(f"Saved to: {output_path}")
    return str(output_path)


if __name__ == '__main__':
    fire.Fire(main)
