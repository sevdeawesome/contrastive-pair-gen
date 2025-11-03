#!/usr/bin/env python3
"""
Train steering vectors for contrastive pairs.

Usage:
    python train_vector.py --contrast-type tom
    python train_vector.py --contrast-type irony --layers 10,20,30 --model google/gemma-2-9b-it
    python train_vector.py --contrast-type self_other --num-samples 100 --output-dir ./my_vectors/
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

# Defer heavy imports until needed (after arg parsing)
# This allows --help to work even if dependencies aren't installed

# Map short names to file names and data extraction functions
CONTRAST_TYPES = {
    'tom': {
        'file': 'theory_of_mind.json',
        'name': 'Theory of Mind',
        'extract': lambda item: (
            item['scenario'] + " " + item['high_tom'],
            item['scenario'] + " " + item['low_tom']
        )
    },
    'irony': {
        'file': 'irony.json',
        'name': 'Irony',
        'extract': lambda item: (item['literal'], item['ironic'])
    },
    'self_other': {
        'file': 'self_other.json',
        'name': 'Self-Other',
        'extract': lambda item: (item['self_subject'], item['other_subject'])
    },
    'harmfulness': {
        'file': 'harmfulness.json',
        'name': 'Harmfulness',
        'extract': lambda item: (
            item['instruction'] + " " + item['harmless'],
            item['instruction'] + " " + item['harmful']
        )
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train steering vectors for contrastive pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ToM vector with default settings (all layers)
  python train_vector.py --contrast-type tom

  # Train on specific layers
  python train_vector.py --contrast-type irony --layers 10,20,30

  # Use different model
  python train_vector.py --contrast-type self_other --model google/gemma-2-9b-it

  # Limit training samples and specify output directory
  python train_vector.py --contrast-type harmfulness --num-samples 100 --output-dir ./my_vectors/

  # Use negative layer indices (count from end)
  python train_vector.py --contrast-type tom --layers=-1,-2,-3
        """
    )

    parser.add_argument(
        '--contrast-type',
        type=str,
        required=True,
        choices=list(CONTRAST_TYPES.keys()),
        help='Type of contrast pairs to use for training'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-27b-it',
        help='HuggingFace model name (default: google/gemma-2-27b-it)'
    )

    parser.add_argument(
        '--layers',
        type=str,
        default='all',
        help='Comma-separated layer numbers (e.g., "10,20,30" or "-1,-2,-3") or "all" (default: all)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for training (default: 2)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of training samples to use (default: all available)'
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
        help='Directory containing contrast pair JSON files (default: auto-detect from script location)'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar during training'
    )

    return parser.parse_args()


def load_contrast_pairs(data_dir, contrast_type, num_samples=None):
    """Load contrast pairs from JSON file"""
    config = CONTRAST_TYPES[contrast_type]
    file_path = Path(data_dir) / config['file']

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find contrast pairs file: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    pairs = [config['extract'](item) for item in data]

    if num_samples is not None:
        pairs = pairs[:num_samples]

    return pairs, config['name']


def parse_layers(layers_str):
    """Parse layer specification string into list of integers or None"""
    if layers_str.lower() == 'all':
        return None  # None means use all layers in steering_vectors library
    else:
        # Support both positive and negative indices
        return [int(x.strip()) for x in layers_str.split(',')]


def get_model_short_name(model_name):
    """Convert full model name to short version for directory name"""
    # Extract key parts: google/gemma-2-27b-it -> gemma-2-27b-it
    # Keep -it/-pt to distinguish instruction-tuned from pretrained
    parts = model_name.split('/')[-1]  # Remove org prefix
    parts = parts.replace('-instruct', '-it').replace('google-', '')
    return parts


def generate_filename(contrast_type, layers, num_samples, date_str):
    """Generate descriptive filename based on configuration (without model name)"""
    if layers is None:
        layer_str = "all-layers"
    else:
        layer_str = "layers-" + "-".join(map(str, layers))

    return f"{contrast_type}_{layer_str}_trained-on-{num_samples}_{date_str}"


def save_steering_vector(steering_vec, output_path, metadata):
    """
    Save steering vector as:
    1. Full object as .pkl (for easy reloading)
    2. Metadata as .json (for reference)
    """
    import pickle

    base_path = output_path.with_suffix('')

    # Save full object as pickle
    pkl_path = base_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(steering_vec, f)
    print(f"  Saved: {pkl_path}")

    # Save metadata as JSON sidecar
    metadata_path = base_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")

    return pkl_path


def main():
    args = parse_args()

    # Import heavy dependencies only after arg parsing
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering_vectors import train_steering_vector

    print("="*70)
    print("Steering Vector Training Pipeline")
    print("="*70)

    # Determine data directory
    if args.data_dir is None:
        # Auto-detect: assume script is in experiments/, data is in ../data/contrast_pairs/
        script_dir = Path(__file__).parent
        args.data_dir = script_dir.parent / 'data' / 'contrast_pairs'
    else:
        args.data_dir = Path(args.data_dir)

    print(f"\nConfiguration:")
    print(f"  Contrast type: {args.contrast_type}")
    print(f"  Model: {args.model}")
    print(f"  Layers: {args.layers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")

    # Load contrast pairs
    print(f"\n[1/4] Loading contrast pairs...")
    pairs, contrast_name = load_contrast_pairs(
        args.data_dir,
        args.contrast_type,
        args.num_samples
    )
    print(f"  ✓ Loaded {len(pairs)} pairs for '{contrast_name}'")

    # Load model and tokenizer
    print(f"\n[2/4] Loading model: {args.model}")
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

    # Parse layers
    layers = parse_layers(args.layers)
    if layers is None:
        print(f"  Training on ALL {num_layers} layers")
    else:
        # Convert negative indices to positive
        layers_display = [l if l >= 0 else num_layers + l for l in layers]
        print(f"  Training on layers: {layers} (absolute: {layers_display})")

    # Train steering vector
    print(f"\n[3/4] Training steering vector...")
    print(f"  Processing {len(pairs)} contrast pairs with batch size {args.batch_size}")
    steering_vec = train_steering_vector(
        model,
        tokenizer,
        pairs,
        layers=layers,
        batch_size=args.batch_size,
        show_progress=not args.no_progress
    )
    print("  ✓ Training complete!")

    # Save steering vector
    print(f"\n[4/4] Saving steering vector...")

    # Create model-specific subdirectory
    model_short = get_model_short_name(args.model)
    model_output_dir = Path(args.output_dir) / model_short
    model_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    filename = generate_filename(args.contrast_type, layers, len(pairs), date_str)
    output_path = model_output_dir / f"{filename}.pkl"

    metadata = {
        'contrast_type': args.contrast_type,
        'contrast_name': contrast_name,
        'model': args.model,
        'model_short': model_short,
        'num_layers': num_layers,
        'layers': layers if layers else 'all',
        'num_samples': len(pairs),
        'batch_size': args.batch_size,
        'timestamp': timestamp,
        'date_trained': datetime.now().isoformat(),
        'filename': filename
    }

    saved_path = save_steering_vector(steering_vec, output_path, metadata)

    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"{'='*70}")
    print(f"\nModel: {args.model}")
    print(f"Trained vector: {saved_path.name}")
    print(f"Location: {saved_path.parent}/")
    print(f"\nTo use this vector:")
    print(f"  import pickle")
    print(f"  with open('{saved_path}', 'rb') as f:")
    print(f"      steering_vec = pickle.load(f)")
    print(f"  with steering_vec.apply(model, multiplier=0.3):")
    print(f"      # generate text here")
    print()


if __name__ == '__main__':
    main()
