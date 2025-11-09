#!/usr/bin/env python3
"""
Evaluate a steering vector on a benchmark dataset.

Usage:
    # Single dataset
    python eval_steering_vector.py \
        google/gemma-2-7b-it \
        ./trained_vectors/gemma-2-7b-it/tom_all-layers_trained-on-92_20251030.pkl \
        0.5 \
        150 \
        simpletom_behavior_qa.json

    # All datasets
    python eval_steering_vector.py \
        google/gemma-2-7b-it \
        ./trained_vectors/gemma-2-7b-it/tom_all-layers_trained-on-92_20251030.pkl \
        0.5 \
        150 \
        all
"""
import json
import pickle
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire


def evaluate_single_dataset(
    dataset_path: Path,
    model,
    tokenizer,
    steering_vec,
    steering_magnitude: float,
    completion_length: int,
    is_instruct: bool
):
    """Evaluate on a single dataset and return results."""
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    dataset_name = dataset_path.name
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"Examples: {len(dataset)}")
    print(f"{'='*70}")

    results = []
    for i, example in enumerate(dataset):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(dataset)}")

        prompt = example['prompt']

        # Format prompt
        if is_instruct:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Generate with steering
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            with steering_vec.apply(model, multiplier=steering_magnitude):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=completion_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

        # Decode output
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Store result
        result = example.copy()
        result['generated_completion'] = generated_text
        results.append(result)

    return results, dataset_name


def main(
    model_name: str,
    intervention_path: str,
    steering_magnitude: float,
    completion_length: int,
    dataset_name: str
):
    """
    Evaluate steering vector on dataset(s).

    Args:
        model_name: HuggingFace model name (e.g., google/gemma-2-7b-it)
        intervention_path: Path to steering vector .pkl file
        steering_magnitude: Multiplier for steering vector (-1.0 to 1.0)
        completion_length: Number of tokens to generate
        dataset_name: Name of JSON file in ./data/eval_data/ or "all" for all datasets
    """
    eval_data_dir = Path("/home/snfiel01/contrastive-pair-gen/data/eval_data")

    # Determine which datasets to evaluate
    if dataset_name.lower() == "all":
        dataset_paths = sorted(eval_data_dir.glob("*.json"))
        print(f"Found {len(dataset_paths)} datasets to evaluate")
    else:
        dataset_paths = [eval_data_dir / dataset_name]
        if not dataset_paths[0].exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_paths[0]}")

    print(f"{'='*70}")
    print(f"Evaluation Configuration")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Steering vector: {intervention_path}")
    print(f"Magnitude: {steering_magnitude}")
    print(f"Completion length: {completion_length}")
    print(f"Datasets: {len(dataset_paths)}")
    print(f"{'='*70}")

    # Load model and tokenizer
    print(f"\nLoading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Model loaded")

    # Load steering vector
    print(f"Loading steering vector: {intervention_path}")
    with open(intervention_path, 'rb') as f:
        steering_vec = pickle.load(f)
    print("✓ Steering vector loaded")

    # Determine if model is instruct-tuned
    is_instruct = '-it' in model_name or 'instruct' in model_name.lower()

    # Setup output directory
    intervention_name = Path(intervention_path).stem
    model_short = model_name.split('/')[-1]
    output_base_dir = Path("/home/snfiel01/contrastive-pair-gen/data/results") / model_short / f"{intervention_name}_{steering_magnitude}"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each dataset
    all_output_paths = []
    for dataset_path in dataset_paths:
        results, dataset_filename = evaluate_single_dataset(
            dataset_path,
            model,
            tokenizer,
            steering_vec,
            steering_magnitude,
            completion_length,
            is_instruct
        )

        # Save results
        output_path = output_base_dir / dataset_filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ Saved to: {output_path}")
        all_output_paths.append(output_path)

    # Final summary
    print(f"\n{'='*70}")
    print(f"✓ Evaluation complete!")
    print(f"{'='*70}")
    print(f"Datasets evaluated: {len(dataset_paths)}")
    print(f"Results directory: {output_base_dir}/")
    print(f"\nOutput files:")
    for path in all_output_paths:
        print(f"  - {path.name}")
    print()


if __name__ == "__main__":
    fire.Fire(main)
