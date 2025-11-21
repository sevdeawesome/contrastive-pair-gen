#!/usr/bin/env python3
"""
Grade steering vector results using Claude API.

Usage:
    python grade_results.py <input_file> <output_file>

Example:
    python grade_results.py _1.json _1.out
"""

import json
import sys
import anthropic
import os
from typing import Dict, List


def grade_generation(client: anthropic.Anthropic, prompt: str, generation: str, correct_answer: str) -> int:
    """
    Use Claude API to grade whether a generation selected the correct answer.

    Returns:
        1 if correct, 0 if incorrect
    """
    grading_prompt = f"""You are grading a multiple choice question response.

Question and options:
{prompt}

Correct answer: {correct_answer}

Model's response:
{generation}

Task: Determine if the model's response indicates they selected answer {correct_answer}.

Look for explicit statements like "the answer is {correct_answer}" or reasoning that leads to option {correct_answer}.

Respond with ONLY one word: "CORRECT" if the model selected {correct_answer}, or "INCORRECT" if they selected a different answer or were unclear."""

    message = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=10,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": grading_prompt
            }
        ]
    )

    response_text = message.content[0].text.strip().upper()
    return 1 if "CORRECT" in response_text else 0


def process_results(input_file: str, output_file: str):
    """
    Process a JSON file of steering vector results and grade them.
    """
    # Initialize Claude client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Read input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Prepare output structure
    output_data = []

    print(f"Processing {len(data)} questions...")

    for idx, item in enumerate(data):
        item_id = item['id']
        prompt = item['prompt']
        correct_answer = item['correct_answer']
        generations = item['generations']

        print(f"\nGrading question {idx + 1}/{len(data)}: {item_id}")

        # Grade each magnitude
        scores = {}
        for magnitude, generation in generations.items():
            print(f"  Magnitude {magnitude}...", end=" ")
            score = grade_generation(client, prompt, generation, correct_answer)
            scores[magnitude] = score
            print(f"{'✓' if score == 1 else '✗'}")

        # Store results
        output_data.append({
            'id': item_id,
            'correct_answer': correct_answer,
            'scores': scores
        })

    # Write output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results written to {output_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for magnitude in ["-25", "-10", "0", "10", "25", "50"]:
        total_correct = sum(item['scores'].get(magnitude, 0) for item in output_data)
        total_questions = len(output_data)
        accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        print(f"Magnitude {magnitude:>3}: {total_correct}/{total_questions} correct ({accuracy:.1f}%)")


def main():
    if len(sys.argv) != 3:
        print("Usage: python grade_results.py <input_file> <output_file>")
        print("Example: python grade_results.py _1.json _1.out")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    process_results(input_file, output_file)


if __name__ == "__main__":
    main()
