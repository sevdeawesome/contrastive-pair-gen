#!/usr/bin/env python3
"""
Train multiple steering vectors in batch (loads model once for efficiency).

Usage:
    python train_vectors.py --model google/gemma-3-12b-it --all-contrast-types --all-layers --middle-layers
    python train_vectors.py --model google/gemma-3-12b-pt --contrast-types tom irony --layers all,22-24-26
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from itertools import product

# Import heavy dependencies
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

# Import from train_vector.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_vector import (
    CONTRAST_TYPES,
    load_contrast_pairs,
    get_model_short_name,
    generate_filename,
    save_steering_vector
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch train multiple steering vectors (loads model once)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all contrast types with both all-layers and middle-layers configs
  python train_vectors.py --model google/gemma-3-12b-it --all-contrast-types --all-layers --middle-layers

  # Train specific contrast types with custom layer configs
  python train_vectors.py --model google/gemma-3-12b-pt --contrast-types tom irony --layers all,15-18-21

  # Train everything for both IT and PT models (note: requires running twice)
  python train_vectors.py --model google/gemma-3-12b-it --all-contrast-types --all-layers --middle-layers
  python train_vectors.py --model google/gemma-3-12b-pt --all-contrast-types --all-layers --middle-layers
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='HuggingFace model name (e.g., google/gemma-3-12b-it)'
    )

    # Contrast type selection
    contrast_group = parser.add_mutually_exclusive_group(required=True)
    contrast_group.add_argument(
        '--all-contrast-types',
        action='store_true',
        help='Train on all 4 contrast types (tom, irony, self_other, harmfulness)'
    )
    contrast_group.add_argument(
        '--contrast-types',
        nargs='+',
        choices=list(CONTRAST_TYPES.keys()),
        help='Specific contrast types to train'
    )

    # Layer configuration selection
    layer_group = parser.add_argument_group('layer configurations')
    layer_group.add_argument(
        '--all-layers',
        action='store_true',
        help='Include a training run with all layers'
    )
    layer_group.add_argument(
        '--middle-layers',
        action='store_true',
        help='Include a training run with auto-detected middle layers'
    )
    layer_group.add_argument(
        '--layers',
        type=str,
        help='Comma-separated layer configs (e.g., "all,10-20-30,15-18-21")'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for training (default: 2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./trained_vectors',
        help='Directory to save trained vectors (default: ./trained_vectors)'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing contrast pair JSON files (default: auto-detect)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar during training'
    )

    return parser.parse_args()


def parse_layer_configs(args, num_layers):
    """Parse and create list of layer configurations to train"""
    configs = []

    # Add all-layers config
    if args.all_layers:
        configs.append(('all', None))

    # Add middle-layers config (auto-detect based on model size)
    if args.middle_layers:
        mid_point = num_layers // 2
        middle = [mid_point - 2, mid_point, mid_point + 2]
        configs.append(('middle', middle))

    # Add custom layer configs
    if args.layers:
        for layer_spec in args.layers.split(','):
            layer_spec = layer_spec.strip()
            if layer_spec.lower() == 'all':
                if ('all', None) not in configs:
                    configs.append(('all', None))
            else:
                # Parse like "10-20-30" or "15,18,21"
                layers = [int(x) for x in layer_spec.replace('-', ',').split(',')]
                config_name = f"layers-{'-'.join(map(str, layers))}"
                configs.append((config_name, layers))

    if not configs:
        raise ValueError("Must specify at least one layer configuration (--all-layers, --middle-layers, or --layers)")

    return configs


def main():
    args = parse_args()

    print("="*70)
    print("Batch Steering Vector Training Pipeline")
    print("="*70)

    # Determine contrast types
    if args.all_contrast_types:
        contrast_types = list(CONTRAST_TYPES.keys())
    else:
        contrast_types = args.contrast_types

    print(f"\n📋 Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Contrast types: {', '.join(contrast_types)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")

    # Determine data directory
    if args.data_dir is None:
        script_dir = Path(__file__).parent
        args.data_dir = script_dir.parent / 'data' / 'contrast_pairs'
    else:
        args.data_dir = Path(args.data_dir)

    # Load all contrast pairs
    print(f"\n[Step 1/{3}] Loading all contrast pair datasets...")
    all_pairs = {}
    for contrast_type in contrast_types:
        pairs, contrast_name = load_contrast_pairs(
            args.data_dir,
            contrast_type,
            num_samples=None  # Always use all samples
        )
        all_pairs[contrast_type] = {
            'pairs': pairs,
            'name': contrast_name,
            'count': len(pairs)
        }
        print(f"  ✓ {contrast_name}: {len(pairs)} pairs")

    # Load model and tokenizer ONCE
    print(f"\n[Step 2/{3}] Loading model: {args.model}")
    print(f"  This may take a minute...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Handle different config structures (Gemma 2 vs Gemma 3)
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        raise ValueError(f"Could not determine number of layers for model {args.model}")

    print(f"  ✓ Model loaded successfully!")
    print(f"  Model has {num_layers} layers")

    # Parse layer configurations
    layer_configs = parse_layer_configs(args, num_layers)
    print(f"\n  Layer configurations to train:")
    for config_name, layers in layer_configs:
        if layers is None:
            print(f"    • {config_name}: all {num_layers} layers")
        else:
            print(f"    • {config_name}: {layers}")

    # Calculate total number of vectors to train
    total_vectors = len(contrast_types) * len(layer_configs)
    print(f"\n  📊 Total vectors to train: {total_vectors}")
    print(f"     ({len(contrast_types)} contrast types × {len(layer_configs)} layer configs)")

    # Train all combinations
    print(f"\n[Step 3/{3}] Training {total_vectors} steering vectors...")
    print("="*70)

    model_short = get_model_short_name(args.model)
    model_output_dir = Path(args.output_dir) / model_short
    model_output_dir.mkdir(parents=True, exist_ok=True)

    trained_count = 0
    for contrast_type, (config_name, layers) in product(contrast_types, layer_configs):
        trained_count += 1
        contrast_data = all_pairs[contrast_type]

        print(f"\n[{trained_count}/{total_vectors}] Training: {contrast_data['name']} ({config_name})")
        print(f"  Pairs: {contrast_data['count']}")
        if layers is None:
            print(f"  Layers: all {num_layers} layers")
        else:
            print(f"  Layers: {layers}")

        # Train steering vector
        steering_vec = train_steering_vector(
            model,
            tokenizer,
            contrast_data['pairs'],
            layers=layers,
            batch_size=args.batch_size,
            show_progress=not args.no_progress
        )

        # Save steering vector
        date_str = datetime.now().strftime("%Y%m%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate filename based on config_name or layers
        if config_name in ['all', 'middle']:
            if config_name == 'all':
                filename = generate_filename(contrast_type, None, contrast_data['count'], date_str)
            else:  # middle
                filename = generate_filename(contrast_type, layers, contrast_data['count'], date_str)
        else:
            filename = generate_filename(contrast_type, layers, contrast_data['count'], date_str)

        output_path = model_output_dir / f"{filename}.pkl"

        metadata = {
            'contrast_type': contrast_type,
            'contrast_name': contrast_data['name'],
            'model': args.model,
            'model_short': model_short,
            'num_layers': num_layers,
            'layers': layers if layers else 'all',
            'layer_config_name': config_name,
            'num_samples': contrast_data['count'],
            'batch_size': args.batch_size,
            'timestamp': timestamp,
            'date_trained': datetime.now().isoformat(),
            'filename': filename
        }

        save_steering_vector(steering_vec, output_path, metadata)
        print(f"  ✓ Saved: {filename}.pkl")

    print(f"\n{'='*70}")
    print(f"✓ Batch training complete!")
    print(f"{'='*70}")
    print(f"\nTrained {total_vectors} steering vectors:")
    print(f"  Model: {args.model}")
    print(f"  Contrast types: {', '.join(contrast_types)}")
    print(f"  Layer configs: {len(layer_configs)}")
    print(f"  Location: {model_output_dir}/")
    print()


if __name__ == '__main__':
    main()
