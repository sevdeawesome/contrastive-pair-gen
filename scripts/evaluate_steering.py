#!/usr/bin/env python3
"""
Evaluate a steering vector on a dataset.

Usage:
    python evaluate_steering.py <model_name> <intervention_path> <steering_magnitude> <completion_length> <dataset_name>

Example:
    python evaluate_steering.py google/gemma-2-7b-it ./trained_vectors/gemma-2-7b-it/tom_all-layers.pkl 1.0 100 self_other_prompts.json
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire


def main(model_name, intervention_path, steering_magnitude, completion_length, dataset_name):
    """
    Evaluate a steering vector on a dataset.

    Args:
        model_name: HuggingFace model name (e.g., 'google/gemma-2-7b-it')
        intervention_path: Path to steering vector pickle file
        steering_magnitude: Multiplier for steering vector (float)
        completion_length: Number of tokens to generate (int)
        dataset_name: Name of JSON file in ./data/eval_data/
    """
    print("="*70)
    print("Steering Vector Evaluation")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Steering vector: {intervention_path}")
    print(f"Magnitude: {steering_magnitude}")
    print(f"Completion length: {completion_length}")
    print(f"Dataset: {dataset_name}")

    # Load model and tokenizer
    print(f"\n[1/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Model loaded")

    # Load steering vector
    print(f"\n[2/5] Loading steering vector...")
    with open(intervention_path, 'rb') as f:
        steering_vec = pickle.load(f)
    print("  ✓ Steering vector loaded")

    # Load dataset
    print(f"\n[3/5] Loading dataset...")
    dataset_path = Path('./data/eval_data') / dataset_name
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    print(f"  ✓ Loaded {len(dataset)} items")

    # Run evaluation
    print(f"\n[4/5] Generating completions...")
    results = []

    for i, item in enumerate(dataset):
        # Extract prompt (handle different formats)
        if 'prompt' in item:
            prompt = item['prompt']
        elif 'text' in item:
            prompt = item['text']
        else:
            prompt = str(item)

        # Generate baseline
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            baseline_outputs = model.generate(
                **inputs,
                max_new_tokens=completion_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        baseline_text = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)

        # Generate with steering
        with steering_vec.apply(model, multiplier=steering_magnitude):
            with torch.no_grad():
                steered_outputs = model.generate(
                    **inputs,
                    max_new_tokens=completion_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
        steered_text = tokenizer.decode(steered_outputs[0], skip_special_tokens=True)

        # Store results
        result = {
            'id': item.get('id', i),
            'prompt': prompt,
            'baseline_completion': baseline_text,
            'steered_completion': steered_text,
            'original_item': item
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} items")

    print(f"  ✓ Completed {len(results)} evaluations")

    # Save results
    print(f"\n[5/5] Saving results...")
    model_short = model_name.split('/')[-1]
    vector_name = Path(intervention_path).stem
    output_dir = Path('./data/results') / model_short / f"{vector_name}_{steering_magnitude}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / dataset_name

    output_data = {
        'metadata': {
            'model': model_name,
            'intervention_path': intervention_path,
            'steering_magnitude': steering_magnitude,
            'completion_length': completion_length,
            'dataset': dataset_name,
            'num_items': len(results),
            'timestamp': datetime.now().isoformat()
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  ✓ Results saved to: {output_path}")
    print(f"\n{'='*70}")
    print("✓ Evaluation complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    fire.Fire(main)
