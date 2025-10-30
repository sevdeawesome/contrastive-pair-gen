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

# Get BOTH ironic and non-ironic examples
ironic_df = df[df['label'] == 1].copy()
ironic_df = ironic_df[ironic_df['comment_text'] != '[deleted]']

non_ironic_df = df[df['label'] == -1].copy()
non_ironic_df = non_ironic_df[non_ironic_df['comment_text'] != '[deleted]']

print(f"Found {len(ironic_df)} ironic examples")
print(f"Found {len(non_ironic_df)} non-ironic examples")

def generate_literal_from_ironic(ironic_text):
    """
    Generate a DIRECT, NON-SARCASTIC version of an ironic statement.
    """
    prompt = f"""You are creating contrast pairs for training AI steering vectors.

Given this IRONIC/SARCASTIC statement, generate a LITERAL/DIRECT version.

IRONIC STATEMENT:
"{ironic_text}"

Requirements for the LITERAL version:
1. Say the SAME thing but DIRECTLY - no sarcasm, no irony
2. If the irony uses OPPOSITE meaning (e.g., "Great job!" when something failed), state it DIRECTLY
3. If the irony uses EXAGGERATION (e.g., "Oh sure, that'll fix everything"), state it PLAINLY
4. Keep similar length and structure
5. Remove all sarcastic markers: "Oh sure...", "Yeah right...", "Great...", ellipses "...", etc.

Examples:
- IRONIC: "Yeah, that'll definitely work..." → LITERAL: "That probably won't work."
- IRONIC: "Oh sure, climate change is totally fake." → LITERAL: "Climate change is real."
- IRONIC: "Great job breaking the vase!" → LITERAL: "You broke the vase and that's problematic."

Generate ONLY the literal version. No explanation.

LITERAL VERSION:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You convert sarcastic statements to direct, literal ones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )

        literal = response.choices[0].message.content.strip()
        # Remove quotes if present
        if literal.startswith('"') and literal.endswith('"'):
            literal = literal[1:-1]
        return literal
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_ironic_from_literal(literal_text):
    """
    Generate a CLEARLY IRONIC/SARCASTIC version of a literal statement.
    """
    prompt = f"""You are creating contrast pairs for training AI steering vectors.

Given this LITERAL/DIRECT statement, generate an IRONIC/SARCASTIC version.

LITERAL STATEMENT:
"{literal_text}"

Requirements for the IRONIC version:
1. Use CLEAR sarcasm markers: "Oh sure...", "Yeah right...", "Great...", "Wonderful...", etc.
2. Use OPPOSITE meaning or EXAGGERATION to convey sarcasm
3. Add ellipses "..." for sarcastic emphasis
4. Make it OBVIOUSLY sarcastic - no ambiguity
5. Keep similar length and structure

Examples:
- LITERAL: "That probably won't work." → IRONIC: "Oh sure, that'll definitely work..."
- LITERAL: "Climate change is real." → IRONIC: "Yeah right, climate change is totally fake..."
- LITERAL: "You broke the vase." → IRONIC: "Great job breaking the vase!"

Generate ONLY the ironic version. No explanation.

IRONIC VERSION:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You convert direct statements to clearly sarcastic, ironic ones."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        ironic = response.choices[0].message.content.strip()
        # Remove quotes if present
        if ironic.startswith('"') and ironic.endswith('"'):
            ironic = ironic[1:-1]
        return ironic
    except Exception as e:
        print(f"Error: {e}")
        return None

def has_clear_irony_markers(text):
    """
    Check if text has clear ironic/sarcastic markers.
    """
    irony_markers = [
        'oh sure', 'yeah right', 'oh great', 'wonderful', 'brilliant',
        'totally', 'definitely', 'absolutely', 'shocking', 'amazing',
        '...', 'lol', 'haha', '/s'
    ]

    text_lower = text.lower()

    # Check for markers
    has_marker = any(marker in text_lower for marker in irony_markers)

    # Check for excessive punctuation (!!!, ???)
    has_emphasis = '!!!' in text or '???' in text or text.count('!') >= 2

    # Check for all caps words (often sarcastic)
    words = text.split()
    has_caps = any(word.isupper() and len(word) > 2 for word in words)

    return has_marker or has_emphasis or has_caps

def validate_pair_quality(ironic, literal):
    """
    Strict validation for clear irony contrast.
    """
    if not ironic or not literal:
        return False, "Empty text"

    # Length check
    len_ratio = max(len(ironic), len(literal)) / min(len(ironic), len(literal))
    if len_ratio > 2.5:
        return False, "Length difference too large"

    # Ironic version MUST have clear markers
    if not has_clear_irony_markers(ironic):
        return False, "No clear irony markers in ironic version"

    # Literal version should NOT have irony markers
    if has_clear_irony_markers(literal):
        return False, "Literal version has irony markers"

    # Should not be identical
    if ironic.lower() == literal.lower():
        return False, "Texts are identical"

    # Basic similarity check (should share some words)
    ironic_words = set(ironic.lower().split())
    literal_words = set(literal.lower().split())
    overlap = len(ironic_words & literal_words) / max(len(ironic_words), len(literal_words))

    if overlap < 0.2:
        return False, f"Too dissimilar (overlap: {overlap:.2f})"

    return True, "Valid"

# Generate pairs
all_pairs = []

print("\n" + "="*60)
print("GENERATING LITERAL VERSIONS FROM IRONIC EXAMPLES")
print("="*60)

# Filter for good ironic examples (with clear sarcasm)
good_ironic = ironic_df[ironic_df['comment_text'].str.len() > 30]
good_ironic = good_ironic[good_ironic['comment_text'].str.len() < 300]
sampled_ironic = good_ironic.sample(n=min(75, len(good_ironic)), random_state=42)

for idx, row in sampled_ironic.iterrows():
    if len(all_pairs) >= 75:
        break

    ironic_text = row['comment_text']

    # Only process if it has clear irony markers
    if not has_clear_irony_markers(ironic_text):
        continue

    print(f"\n[{len(all_pairs)+1}/75] Processing ironic example...")
    print(f"  IRONIC: {ironic_text[:100]}...")

    literal_text = generate_literal_from_ironic(ironic_text)

    if literal_text:
        is_valid, reason = validate_pair_quality(ironic_text, literal_text)
        if is_valid:
            all_pairs.append({"ironic": ironic_text, "literal": literal_text})
            print(f"  LITERAL: {literal_text[:100]}...")
            print(f"  ✓ VALID")
        else:
            print(f"  ✗ REJECTED: {reason}")

    time.sleep(0.3)

print(f"\nGenerated {len(all_pairs)} pairs from ironic examples")

print("\n" + "="*60)
print("GENERATING IRONIC VERSIONS FROM LITERAL EXAMPLES")
print("="*60)

# Filter for good non-ironic examples
good_literal = non_ironic_df[non_ironic_df['comment_text'].str.len() > 30]
good_literal = good_literal[good_literal['comment_text'].str.len() < 300]
# Exclude ones that already have irony markers
good_literal = good_literal[~good_literal['comment_text'].apply(has_clear_irony_markers)]
sampled_literal = good_literal.sample(n=min(75, len(good_literal)), random_state=42)

target_pairs = 150 - len(all_pairs)

for idx, row in sampled_literal.iterrows():
    if len(all_pairs) >= 150:
        break

    literal_text = row['comment_text']

    print(f"\n[{len(all_pairs)+1}/150] Processing literal example...")
    print(f"  LITERAL: {literal_text[:100]}...")

    ironic_text = generate_ironic_from_literal(literal_text)

    if ironic_text:
        is_valid, reason = validate_pair_quality(ironic_text, literal_text)
        if is_valid:
            all_pairs.append({"ironic": ironic_text, "literal": literal_text})
            print(f"  IRONIC: {ironic_text[:100]}...")
            print(f"  ✓ VALID")
        else:
            print(f"  ✗ REJECTED: {reason}")

    time.sleep(0.3)

print(f"\n\n" + "="*60)
print(f"GENERATION COMPLETE")
print("="*60)
print(f"Total valid pairs: {len(all_pairs)}")

# Save
output_path = "/home/sevdeawesome/projects/contrastive-pair-gen/irony_pairs.json"
with open(output_path, 'w') as f:
    json.dump(all_pairs, f, indent=2)

print(f"\n✓ Saved to: {output_path}")

# Show samples
print("\n" + "="*60)
print("SAMPLE PAIRS FOR REVIEW")
print("="*60)

for i in range(min(10, len(all_pairs))):
    print(f"\n{i+1}.")
    print(f"  IRONIC:  {all_pairs[i]['ironic']}")
    print(f"  LITERAL: {all_pairs[i]['literal']}")
