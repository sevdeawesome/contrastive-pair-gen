import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def validate_pair(ironic_text, literal_text):
    """
    Use GPT-4 to validate that one text is ironic and the other is literal,
    and that they convey similar information.

    Returns: (is_valid, reason)
    """
    prompt = f"""You are validating contrast pairs for training steering vectors.

Evaluate whether these two texts form a valid ironic/literal contrast pair:

TEXT 1: {ironic_text}

TEXT 2: {literal_text}

A valid pair must meet ALL these criteria:
1. One text is clearly IRONIC/SARCASTIC (using sarcasm, exaggeration, understatement, or opposite meaning)
2. One text is clearly LITERAL/SINCERE (straightforward, no sarcasm)
3. Both texts convey similar core information/meaning
4. The primary difference is TONE (ironic vs literal), not content
5. They are roughly similar in length and structure

Respond with ONLY a JSON object in this exact format:
{{"valid": true/false, "reason": "brief explanation", "ironic_is": "text1" or "text2"}}

Examples:
- Valid: "Great job breaking the vase!" (ironic) vs "You broke the vase" (literal)
- Invalid: "I love pizza" (literal) vs "Dogs are great" (literal) - different topics
- Invalid: "Nice weather" (could be either) vs "It's raining" (unclear contrast)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for better validation
            messages=[
                {"role": "system", "content": "You are a precise validator that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent validation
            max_tokens=150
        )

        result_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)

        return result['valid'], result.get('reason', 'No reason provided')

    except Exception as e:
        print(f"Validation error: {e}")
        return None, str(e)

# Load both datasets
print("Loading datasets...")
with open('/home/sevdeawesome/projects/contrastive-pair-gen/irony/irony_pairs_raw.json', 'r') as f:
    original_pairs = json.load(f)

with open('/home/sevdeawesome/projects/contrastive-pair-gen/irony/irony_pairs_reverse_raw.json', 'r') as f:
    reverse_pairs = json.load(f)

print(f"Loaded {len(original_pairs)} original pairs (ironic → literal)")
print(f"Loaded {len(reverse_pairs)} reverse pairs (literal → ironic)")

# Normalize keys (original uses "non_ironic", reverse uses "literal")
for pair in original_pairs:
    if 'non_ironic' in pair:
        pair['literal'] = pair.pop('non_ironic')

# Combine datasets
all_pairs = original_pairs + reverse_pairs
print(f"\nTotal pairs to validate: {len(all_pairs)}")

# Validate all pairs
validated_pairs = []
rejected_pairs = []

print("\nValidating pairs...")
for i, pair in enumerate(all_pairs):
    print(f"\nValidating {i+1}/{len(all_pairs)}...")
    print(f"  Ironic:  {pair['ironic'][:80]}...")
    print(f"  Literal: {pair['literal'][:80]}...")

    is_valid, reason = validate_pair(pair['ironic'], pair['literal'])

    if is_valid is None:
        print(f"  ⚠ Validation failed: {reason}")
        rejected_pairs.append({**pair, 'rejection_reason': reason})
    elif is_valid:
        print(f"  ✓ VALID: {reason}")
        validated_pairs.append(pair)
    else:
        print(f"  ✗ REJECTED: {reason}")
        rejected_pairs.append({**pair, 'rejection_reason': reason})

    # Rate limiting
    time.sleep(0.3)

print(f"\n\n{'='*60}")
print(f"VALIDATION COMPLETE")
print(f"{'='*60}")
print(f"Valid pairs: {len(validated_pairs)}/{len(all_pairs)} ({100*len(validated_pairs)/len(all_pairs):.1f}%)")
print(f"Rejected pairs: {len(rejected_pairs)}/{len(all_pairs)} ({100*len(rejected_pairs)/len(all_pairs):.1f}%)")

# Save validated pairs to main directory
output_path = '/home/sevdeawesome/projects/contrastive-pair-gen/irony_pairs.json'
with open(output_path, 'w') as f:
    json.dump(validated_pairs, f, indent=2)

print(f"\n✓ Saved validated dataset to: {output_path}")

# Save rejected pairs for review
rejected_path = '/home/sevdeawesome/projects/contrastive-pair-gen/irony/rejected_pairs.json'
with open(rejected_path, 'w') as f:
    json.dump(rejected_pairs, f, indent=2)

print(f"✓ Saved rejected pairs to: {rejected_path}")

# Statistics
print(f"\n{'='*60}")
print(f"FINAL DATASET STATISTICS")
print(f"{'='*60}")
print(f"Total validated pairs: {len(validated_pairs)}")

ironic_lengths = [len(p['ironic']) for p in validated_pairs]
literal_lengths = [len(p['literal']) for p in validated_pairs]

print(f"Average ironic length: {sum(ironic_lengths) / len(ironic_lengths):.1f} chars")
print(f"Average literal length: {sum(literal_lengths) / len(literal_lengths):.1f} chars")
print(f"Average length difference: {sum(abs(i-l) for i, l in zip(ironic_lengths, literal_lengths)) / len(validated_pairs):.1f} chars")

# Show sample validated pairs
print(f"\n{'='*60}")
print(f"SAMPLE VALIDATED PAIRS")
print(f"{'='*60}")
for i in range(min(5, len(validated_pairs))):
    print(f"\n{i+1}. IRONIC:  {validated_pairs[i]['ironic']}")
    print(f"   LITERAL: {validated_pairs[i]['literal']}")
