"""
Train and save steering vectors for all contrast pair types.
This script should be run once (or when you update your training data).
The saved vectors can then be quickly loaded by the chat interface.
"""

import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

# Configuration
MODEL_NAME = "google/gemma-2-27b-it"
CONTRAST_DATA_DIR = Path("contrast_pair_data")
OUTPUT_DIR = Path("trained_steering_vectors")
OUTPUT_DIR.mkdir(exist_ok=True)

# Training settings
TRAINING_CONFIG = {
    "layers": [22, 24, 26],  # Middle layers for Gemma-2-27B (46 layers total)
    "batch_size": 4,  # Can use larger batch since we're not running the interface yet
    "show_progress": True,
}

def load_contrast_pairs():
    """Load all contrast pair files"""
    pairs = {}

    # Theory of Mind pairs
    tom_path = CONTRAST_DATA_DIR / "tom_pairs.json"
    if tom_path.exists():
        with open(tom_path, 'r') as f:
            tom_data = json.load(f)
            pairs['theory_of_mind'] = [
                (item['scenario'] + " " + item['high_tom'],
                 item['scenario'] + " " + item['low_tom'])
                for item in tom_data
            ]
            print(f"✓ Loaded {len(pairs['theory_of_mind'])} Theory of Mind pairs")
    else:
        print(f"⚠ Skipping Theory of Mind: {tom_path} not found")

    # Self-Other pairs
    self_other_path = CONTRAST_DATA_DIR / "self_other_pairs.json"
    if self_other_path.exists():
        with open(self_other_path, 'r') as f:
            self_other_data = json.load(f)
            pairs['self_other'] = [
                (item['self_subject'], item['other_subject'])
                for item in self_other_data
            ]
            print(f"✓ Loaded {len(pairs['self_other'])} Self-Other pairs")
    else:
        print(f"⚠ Skipping Self-Other: {self_other_path} not found")

    # Irony pairs
    irony_path = CONTRAST_DATA_DIR / "irony_pairs_2.json"
    if irony_path.exists():
        with open(irony_path, 'r') as f:
            irony_data = json.load(f)
            pairs['irony'] = [
                (item['literal'], item['ironic'])  # Positive = literal, Negative = ironic
                for item in irony_data
            ]
            print(f"✓ Loaded {len(pairs['irony'])} Irony pairs")
    else:
        print(f"⚠ Skipping Irony: {irony_path} not found")

    # Harmfulness pairs
    harm_path = CONTRAST_DATA_DIR / "harmfulness_pairs.json"
    if harm_path.exists():
        with open(harm_path, 'r') as f:
            harm_data = json.load(f)
            pairs['harmfulness'] = [
                (item['instruction'] + " " + item['harmless'],
                 item['instruction'] + " " + item['harmful'])
                for item in harm_data
            ]
            print(f"✓ Loaded {len(pairs['harmfulness'])} Harmfulness pairs")
    else:
        print(f"⚠ Skipping Harmfulness: {harm_path} not found")

    return pairs


def main():
    print("="*70)
    print("STEERING VECTOR TRAINING SCRIPT")
    print("="*70)

    # Load model and tokenizer
    print(f"\n[1/3] Loading model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Model loaded successfully!")

    # Load contrast pairs
    print(f"\n[2/3] Loading contrast pairs from {CONTRAST_DATA_DIR}/...")
    contrast_pairs = load_contrast_pairs()

    if not contrast_pairs:
        print("\n❌ No contrast pairs found! Check your data directory.")
        return

    print(f"\n✓ Found {len(contrast_pairs)} contrast types to train")

    # Train and save steering vectors
    print(f"\n[3/3] Training steering vectors...")
    print(f"Config: {TRAINING_CONFIG}\n")

    for name, pairs in contrast_pairs.items():
        print(f"\n{'─'*70}")
        print(f"Training: {name.upper().replace('_', ' ')}")
        print(f"  Pairs: {len(pairs)}")
        print(f"{'─'*70}")

        try:
            steering_vec = train_steering_vector(
                model,
                tokenizer,
                pairs,
                **TRAINING_CONFIG
            )

            # Save the steering vector
            output_path = OUTPUT_DIR / f"{name}.pt"
            torch.save(steering_vec, output_path)
            print(f"✓ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Failed to train {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nSteering vectors saved to: {OUTPUT_DIR}/")
    print("You can now use steering_chat_interface_fast.py for quick inference.")


if __name__ == "__main__":
    main()
