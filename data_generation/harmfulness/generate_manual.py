#!/usr/bin/env python3
"""
Manually generate harmfulness/refusal contrast pairs with variation in response patterns.
This ensures high-quality pairs that meet all criteria and adds variety to capture more variance.
"""

import json
from pathlib import Path
import random

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
HARMBENCH_FILE = SCRIPT_DIR / "harmbench_test.json"
OUTPUT_FILE = PROJECT_DIR / "harmfulness_pairs.json"

# Response templates with variation
HARMFUL_TEMPLATES = [
    "Sure! I can {action}.",
    "I'll {action}.",
    "Absolutely, I can {action}.",
    "Of course, I can {action}.",
    "I'd be happy to {action}.",
    "Certainly, I'll {action}.",
    "Yes, I can {action}.",
    "I can {action}.",
]

HARMLESS_TEMPLATES = [
    "No, I cannot {action}.",
    "I will not {action}.",
    "I refuse to {action}.",
    "Absolutely not, I won't {action}.",
    "I cannot {action}.",
    "I can't {action}.",
    "No, I won't {action}.",
    "I'm unable to {action}.",
]

def extract_action(instruction: str) -> str:
    """Convert instruction to action form suitable for templates."""
    # Just use the full instruction lowercased
    return instruction.lower()

def generate_pair(instruction: str, category: str, index: int) -> dict:
    """Generate a single harmfulness/refusal contrast pair with varied response patterns."""

    action = extract_action(instruction)

    # Use index to deterministically select templates for consistency
    # but with variation across the dataset
    harmful_template = HARMFUL_TEMPLATES[index % len(HARMFUL_TEMPLATES)]
    harmless_template = HARMLESS_TEMPLATES[index % len(HARMLESS_TEMPLATES)]

    harmful = harmful_template.format(action=action)
    harmless = harmless_template.format(action=action)

    return {
        "instruction": instruction,
        "harmful": harmful,
        "harmless": harmless,
        "category": category
    }

def main():
    """Generate all contrast pairs."""

    print("🚀 Generating harmfulness/refusal contrast pairs manually...")
    print(f"📁 Input: {HARMBENCH_FILE}")
    print(f"📁 Output: {OUTPUT_FILE}")
    print()

    # Load HarmBench data
    with open(HARMBENCH_FILE, 'r') as f:
        harmbench_data = json.load(f)

    total = len(harmbench_data)
    print(f"📊 Total instructions: {total}")
    print()

    # Generate pairs
    pairs = []
    for i, item in enumerate(harmbench_data):
        instruction = item["instruction"]
        category = item.get("category", "unknown")

        pair = generate_pair(instruction, category, i)
        pairs.append(pair)

        if (i + 1) % 50 == 0:
            print(f"✅ Generated {i + 1}/{total} pairs...")

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(pairs, f, indent=2)

    print()
    print("=" * 70)
    print(f"✅ Successfully generated {len(pairs)} contrast pairs!")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print()

    # Show some examples
    print("Sample pairs:")
    for i in [0, 50, 100, 150]:
        if i < len(pairs):
            print(f"\n[{i+1}] {pairs[i]['category']}")
            print(f"Instruction: {pairs[i]['instruction'][:70]}...")
            print(f"Harmful:     {pairs[i]['harmful']}")
            print(f"Harmless:    {pairs[i]['harmless']}")

    print("=" * 70)

if __name__ == "__main__":
    main()
