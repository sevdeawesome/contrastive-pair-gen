"""
Download and convert all SimpleToM datasets to JSON format for evaluation.
"""

from datasets import load_dataset
import json
import os

# Create output directory
output_dir = "data/eval_data"
os.makedirs(output_dir, exist_ok=True)

# Define the three dataset configurations
dataset_configs = [
    "behavior-qa",
    "judgment-qa",
    "mental-state-qa"
]

def convert_to_json_format(dataset, dataset_name):
    """Convert dataset to our JSON format with multiple choice."""
    json_data = []

    for idx, example in enumerate(dataset['test']):
        # Build the full prompt with story, question, and choices
        story = example['story']
        question = example['question']

        # Get choices
        choice_texts = example['choices']['text']
        choice_labels = example['choices']['label']

        # Format choices as A) option1, B) option2, etc.
        choices_formatted = []
        for label, text in zip(choice_labels, choice_texts):
            choices_formatted.append(f"{label}) {text}")

        # Create the full prompt
        prompt = f"{story}\n\n{question}\n\n" + "\n".join(choices_formatted)

        # Get the correct answer (both the letter and the full text)
        answer_key = example['answerKey']
        answer_idx = choice_labels.index(answer_key)
        correct_answer_text = choice_texts[answer_idx]

        # Determine ToM type and difficulty from the scenario
        scenario = example['scenario_name']
        tom_type = "false_belief"  # Default

        # Try to infer difficulty from id or scenario
        if 'sev1' in example['id']:
            difficulty = "easy"
        elif 'sev2' in example['id']:
            difficulty = "medium"
        elif 'sev3' in example['id']:
            difficulty = "hard"
        else:
            difficulty = "unknown"

        json_entry = {
            "id": f"simpletom_{dataset_name}_{example['id']}",
            "prompt": prompt,
            "choices": {
                "labels": choice_labels,
                "texts": choice_texts
            },
            "correct_answer": answer_key,
            "correct_answer_text": correct_answer_text,
            "category": dataset_name,
            "metadata": {
                "scenario": scenario,
                "difficulty": difficulty,
                "original_id": example['id']
            }
        }

        json_data.append(json_entry)

    return json_data

# Process each dataset
for config in dataset_configs:
    print(f"\n{'='*60}")
    print(f"Processing {config}...")
    print(f"{'='*60}")

    # Load dataset
    ds = load_dataset("allenai/SimpleToM", config)

    # Show stats
    print(f"Number of examples: {len(ds['test'])}")

    # Convert to JSON format
    json_data = convert_to_json_format(ds, config)

    # Save to file
    output_file = os.path.join(output_dir, f"simpletom_{config.replace('-', '_')}.json")
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved {len(json_data)} examples to {output_file}")

    # Show example
    print(f"\nExample from {config}:")
    print(json.dumps(json_data[0], indent=2))

print(f"\n{'='*60}")
print("All datasets processed successfully!")
print(f"{'='*60}")
