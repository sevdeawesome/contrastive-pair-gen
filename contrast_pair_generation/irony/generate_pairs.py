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

# Filter for ironic examples and remove deleted comments
ironic_df = df[df['label'] == 1].copy()
ironic_df = ironic_df[ironic_df['comment_text'] != '[deleted]']

print(f"Found {len(ironic_df)} ironic examples (excluding deleted)")

def generate_non_ironic_version(ironic_text):
    """
    Use OpenAI to generate a non-ironic version of an ironic statement.
    """
    prompt = f"""You are creating contrast pairs for training steering vectors. Given an IRONIC statement, generate a LITERAL/NON-IRONIC version that:

1. Conveys the same core meaning/information
2. Has the same structure (similar sentence pattern)
3. Is roughly the same length (within ~10 tokens)
4. Changes ONLY the ironic tone to a literal/sincere tone
5. Keeps the same perspective and context

IRONIC STATEMENT:
{ironic_text}

Generate ONLY the non-ironic version. Do not include any explanation or labels. Just output the literal statement.

NON-IRONIC VERSION:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts ironic statements to literal ones while maintaining structure and length."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        non_ironic = response.choices[0].message.content.strip()
        return non_ironic
    except Exception as e:
        print(f"Error generating pair: {e}")
        return None

def validate_pair(ironic, non_ironic):
    """
    Basic validation to ensure the pair meets quality criteria.
    """
    if not non_ironic or len(non_ironic) < 5:
        return False

    # Check length difference (shouldn't be more than 2x different)
    len_ratio = max(len(ironic), len(non_ironic)) / min(len(ironic), len(non_ironic))
    if len_ratio > 2.0:
        return False

    # Should not be identical
    if ironic.lower() == non_ironic.lower():
        return False

    return True

# Generate contrast pairs
contrast_pairs = []
num_pairs_to_generate = 100  # Start with 100 pairs

print(f"\nGenerating {num_pairs_to_generate} contrast pairs...")

# Sample ironic examples
sampled_ironic = ironic_df.sample(n=min(num_pairs_to_generate, len(ironic_df)), random_state=42)

for idx, row in sampled_ironic.iterrows():
    ironic_text = row['comment_text']

    print(f"\nProcessing {len(contrast_pairs) + 1}/{num_pairs_to_generate}...")
    print(f"Ironic: {ironic_text[:100]}...")

    non_ironic_text = generate_non_ironic_version(ironic_text)

    if non_ironic_text and validate_pair(ironic_text, non_ironic_text):
        pair = {
            "ironic": ironic_text,
            "non_ironic": non_ironic_text
        }
        contrast_pairs.append(pair)
        print(f"Non-ironic: {non_ironic_text[:100]}...")
        print("✓ Pair validated")
    else:
        print("✗ Pair rejected")

    # Rate limiting
    time.sleep(0.5)

    if len(contrast_pairs) >= num_pairs_to_generate:
        break

print(f"\n\nGenerated {len(contrast_pairs)} valid contrast pairs")

# Save to JSON
output_path = "/home/sevdeawesome/projects/contrastive-pair-gen/irony/irony_pairs_raw.json"
with open(output_path, 'w') as f:
    json.dump(contrast_pairs, f, indent=2)

print(f"Saved to {output_path}")

# Display some examples
print("\n\n=== SAMPLE PAIRS ===")
for i, pair in enumerate(contrast_pairs[:5]):
    print(f"\nPair {i+1}:")
    print(f"  IRONIC:     {pair['ironic']}")
    print(f"  NON-IRONIC: {pair['non_ironic']}")
