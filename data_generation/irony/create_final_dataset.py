import json
import re

# Load the raw pairs
with open('/home/sevdeawesome/projects/contrastive-pair-gen/irony/irony_pairs_raw.json', 'r') as f:
    raw_pairs = json.load(f)

# Clean and format pairs
final_pairs = []

for pair in raw_pairs:
    ironic = pair['ironic'].strip()
    literal = pair['non_ironic'].strip()

    # Basic cleaning
    # Remove excessive whitespace
    ironic = re.sub(r'\s+', ' ', ironic)
    literal = re.sub(r'\s+', ' ', literal)

    # Skip pairs that are too short
    if len(ironic) < 10 or len(literal) < 10:
        continue

    # Skip pairs where the difference is too large
    len_ratio = max(len(ironic), len(literal)) / min(len(ironic), len(literal))
    if len_ratio > 2.5:
        continue

    # Add to final dataset
    final_pairs.append({
        "ironic": ironic,
        "literal": literal
    })

print(f"Final dataset contains {len(final_pairs)} pairs")

# Save to main directory
output_path = '/home/sevdeawesome/projects/contrastive-pair-gen/irony_pairs.json'
with open(output_path, 'w') as f:
    json.dump(final_pairs, f, indent=2)

print(f"Saved to {output_path}")

# Print statistics
print("\n=== DATASET STATISTICS ===")
print(f"Total pairs: {len(final_pairs)}")

ironic_lengths = [len(p['ironic']) for p in final_pairs]
literal_lengths = [len(p['literal']) for p in final_pairs]

print(f"Average ironic length: {sum(ironic_lengths) / len(ironic_lengths):.1f} chars")
print(f"Average literal length: {sum(literal_lengths) / len(literal_lengths):.1f} chars")
print(f"Average length difference: {sum(abs(i-l) for i, l in zip(ironic_lengths, literal_lengths)) / len(final_pairs):.1f} chars")

# Show some examples
print("\n=== SAMPLE PAIRS ===")
for i in range(min(10, len(final_pairs))):
    print(f"\n{i+1}. IRONIC:")
    print(f"   {final_pairs[i]['ironic'][:150]}...")
    print(f"   LITERAL:")
    print(f"   {final_pairs[i]['literal'][:150]}...")
