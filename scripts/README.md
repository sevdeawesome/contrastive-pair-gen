# Training Scripts

## train_steering_vector.py

Simple CLI for training steering vectors from contrastive pairs.

### Usage

```bash
python scripts/train_steering_vector.py <model_name> <contrast_pair_dataset> <layers> [--batch_size=2]
```

### Parameters

- `model_name`: HuggingFace model name (e.g., `google/gemma-2-7b`, `google/gemma-3-4b-it`)
- `contrast_pair_dataset`: Dataset name without .json extension (e.g., `irony`, `theory_of_mind`, `self_other`, `harmfulness`)
- `layers`: Either `all` or comma-separated layer indices (e.g., `10,20,30`)
- `batch_size`: (Optional) Batch size for training, default=2

### Examples

```bash
# Train on all layers
python scripts/train_steering_vector.py google/gemma-2-7b irony all

# Train on specific layers
python scripts/train_steering_vector.py google/gemma-2-7b theory_of_mind 10,20,30

# Train with custom batch size
python scripts/train_steering_vector.py google/gemma-3-4b-it harmfulness 15,18,21 --batch_size=4
```

### Output

Steering vectors are saved to:
```
./data/trained_vectors/{model_short_name}/{dataset}_{layers}.pkl
```

For example:
- `./data/trained_vectors/gemma-2-7b/irony_all.pkl`
- `./data/trained_vectors/gemma-2-7b/theory_of_mind_10-20-30.pkl`

### Available Datasets

- `theory_of_mind` - ToM contrast pairs (high_tom vs low_tom)
- `irony` - Irony detection (literal vs ironic)
- `self_other` - Self-other distinction (self_subject vs other_subject)
- `harmfulness` - Safety pairs (harmless vs harmful)
