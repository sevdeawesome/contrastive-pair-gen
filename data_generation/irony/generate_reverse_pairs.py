import pandas as pd
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Path to the downloaded dataset
dataset_path = "/home/sevdeawesome/.cache/kagglehub/datasets/rtatman/ironic-corpus/versions/1/irony-labeled.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Filter for NON-ironic examples and remove deleted comments
non_ironic_df = df[df['label'] == -1].copy()
non_ironic_df = non_ironic_df[non_ironic_df['comment_text'] != '[deleted]']

print(f"Found {len(non_ironic_df)} non-ironic examples (excluding deleted)")

def generate_ironic_version(literal_text):
    """
    Use OpenAI to generate an ironic/sarcastic version of a literal statement.
    """
    prompt = f"""You are creating contrast pairs for training steering vectors. Given a LITERAL/SINCERE statement, generate an IRONIC/SARCASTIC version that:

1. Conveys the same core information but with ironic/sarcastic tone
2. Has the same structure (similar sentence pattern)
3. Is roughly the same length (within ~10 tokens)
4. Changes ONLY the tone from literal to ironic/sarcastic
5. Keeps the same perspective and context
6. Uses typical ironic devices: exaggeration, understatement, reversal, or obvious sarcasm

LITERAL STATEMENT:
{literal_text}

Generate ONLY the ironic version. Do not include any explanation or labels. Just output the sarcastic/ironic statement.

IRONIC VERSION:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts literal statements to ironic/sarcastic ones while maintaining structure and length."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # Slightly higher for more creative irony
            max_tokens=200
        )

        ironic = response.choices[0].message.content.strip()
        return ironic
    except Exception as e:
        print(f"Error generating pair: {e}")
        return None

def validate_pair(literal, ironic):
    """
    Basic validation to ensure the pair meets quality criteria.
    """
    if not ironic or len(ironic) < 5:
        return False

    # Check length difference (shouldn't be more than 2x different)
    len_ratio = max(len(literal), len(ironic)) / min(len(literal), len(ironic))
    if len_ratio > 2.0:
        return False

    # Should not be identical
    if literal.lower() == ironic.lower():
        return False

    return True

# Generate contrast pairs (reverse direction)
reverse_pairs = []
num_pairs_to_generate = 100  # Generate 100 more pairs

print(f"\nGenerating {num_pairs_to_generate} reverse contrast pairs (literal → ironic)...")

# Sample non-ironic examples, filtering for reasonable length
filtered_non_ironic = non_ironic_df[non_ironic_df['comment_text'].str.len() > 20]
filtered_non_ironic = filtered_non_ironic[filtered_non_ironic['comment_text'].str.len() < 500]

sampled_non_ironic = filtered_non_ironic.sample(n=min(num_pairs_to_generate * 2, len(filtered_non_ironic)), random_state=42)

for idx, row in sampled_non_ironic.iterrows():
    if len(reverse_pairs) >= num_pairs_to_generate:
        break

    literal_text = row['comment_text']

    print(f"\nProcessing {len(reverse_pairs) + 1}/{num_pairs_to_generate}...")
    print(f"Literal: {literal_text[:100]}...")

    ironic_text = generate_ironic_version(literal_text)

    if ironic_text and validate_pair(literal_text, ironic_text):
        pair = {
            "ironic": ironic_text,
            "literal": literal_text
        }
        reverse_pairs.append(pair)
        print(f"Ironic: {ironic_text[:100]}...")
        print("✓ Pair validated")
    else:
        print("✗ Pair rejected")

    # Rate limiting
    time.sleep(0.5)

print(f"\n\nGenerated {len(reverse_pairs)} valid reverse pairs")

# Save to JSON
output_path = "/home/sevdeawesome/projects/contrastive-pair-gen/irony/irony_pairs_reverse_raw.json"
with open(output_path, 'w') as f:
    json.dump(reverse_pairs, f, indent=2)

print(f"Saved to {output_path}")

# Display some examples
print("\n\n=== SAMPLE REVERSE PAIRS ===")
for i, pair in enumerate(reverse_pairs[:5]):
    print(f"\nPair {i+1}:")
    print(f"  LITERAL: {pair['literal']}")
    print(f"  IRONIC:  {pair['ironic']}")
