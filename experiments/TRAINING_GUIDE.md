# Steering Vector Training Guide

This guide explains how to use `train_vector.py` to train steering vectors from contrast pairs.

## Quick Start

```bash
# Train a Theory of Mind vector with default settings
python experiments/train_vector.py --contrast-type tom

# Train an Irony vector on specific layers
python experiments/train_vector.py --contrast-type irony --layers 22,24,26

# Train with a different model
python experiments/train_vector.py --contrast-type self_other --model google/gemma-2-9b-it

# Save to a specific directory
python experiments/train_vector.py --contrast-type harmfulness --output-dir ./my_vectors/
```

## Command-Line Options

### Required Arguments

- `--contrast-type`: Which type of contrast pairs to train on
  - `tom`: Theory of Mind (false belief tasks)
  - `irony`: Irony vs literal language
  - `self_other`: Self-focused vs other-focused perspectives
  - `harmfulness`: Harmless vs harmful responses

### Optional Arguments

- `--model MODEL`: HuggingFace model identifier
  - Default: `google/gemma-2-27b-it`
  - Example: `google/gemma-2-9b-it`, `meta-llama/Llama-2-7b-chat-hf`

- `--layers LAYERS`: Which layers to train on
  - Default: `all` (trains on all model layers)
  - Can specify comma-separated layer numbers: `10,20,30`
  - Supports negative indices (count from end): `-1,-2,-3`
  - **Tip**: Middle layers often work best. For a 46-layer model, try `22,24,26`

- `--batch-size N`: Batch size for training
  - Default: `2`
  - Larger values are faster but use more memory
  - For 27B models on limited GPU memory, keep at 1-2

- `--num-samples N`: Limit number of training samples
  - Default: Use all available samples
  - Useful for quick testing: `--num-samples 10`

- `--output-dir DIR`: Where to save trained vectors
  - Default: `./trained_vectors/`
  - Directory will be created if it doesn't exist

- `--data-dir DIR`: Custom location for contrast pair JSON files
  - Default: Auto-detects `../data/contrast_pairs/` relative to script
  - Only needed if data is in a non-standard location

- `--no-progress`: Disable progress bar during training
  - Useful for logging or scripted environments

## Output Files

The training script creates **three files** for each trained vector:

1. **`{name}.pkl`** - Full steering vector object (recommended for use)
   - Contains the complete `SteeringVector` object with all methods
   - Load with: `pickle.load(open('file.pkl', 'rb'))`

2. **`{name}.safetensors`** - Just the tensor weights (for safety/sharing)
   - Contains only the raw tensors, no Python objects
   - Safer for sharing or production deployment

3. **`{name}.json`** - Metadata about the training run
   - Model name, layers used, number of samples, timestamp, etc.

### Filename Convention

Files are automatically named based on configuration:

```
{contrast_type}_{model_short}_{layers}_{date}.{ext}

Examples:
- tom_gemma-2-27b_all-layers_20251030.pkl
- irony_gemma-2-9b_layers-22-24-26_20251030.pkl
- self_other_llama-2-7b-chat_layers--1--2--3_20251030.pkl
```

## Usage Examples

### Example 1: Train on all contrast types

```bash
# Train each type separately
for contrast in tom irony self_other harmfulness; do
    python experiments/train_vector.py \
        --contrast-type $contrast \
        --output-dir ./vectors/
done
```

### Example 2: Test different layer configurations

```bash
# Test early layers
python experiments/train_vector.py --contrast-type tom --layers 5,10,15

# Test middle layers (often best)
python experiments/train_vector.py --contrast-type tom --layers 20,25,30

# Test late layers
python experiments/train_vector.py --contrast-type tom --layers -10,-5,-1
```

### Example 3: Quick test with limited data

```bash
# Train on just 50 samples for quick testing
python experiments/train_vector.py \
    --contrast-type irony \
    --num-samples 50 \
    --layers 22,24,26 \
    --output-dir ./test_vectors/
```

### Example 4: Train for different model sizes

```bash
# Small model (9B parameters)
python experiments/train_vector.py \
    --contrast-type tom \
    --model google/gemma-2-9b-it \
    --layers 15,18,21 \
    --batch-size 4

# Large model (27B parameters)
python experiments/train_vector.py \
    --contrast-type tom \
    --model google/gemma-2-27b-it \
    --layers 22,24,26 \
    --batch-size 2
```

## Loading and Using Trained Vectors

Once you've trained a vector, use it like this:

```python
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the steering vector
with open('./trained_vectors/tom_gemma-2-27b_layers-22-24-26_20251030.pkl', 'rb') as f:
    steering_vec = pickle.load(f)

# Load your model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")

# Generate with steering
prompt = "Your prompt here"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

# Apply steering during generation
with steering_vec.apply(model, multiplier=0.5):
    outputs = model.generate(**inputs, max_new_tokens=150)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### Steering Multipliers

- **Positive values (0.1 to 1.0)**: Enhance the capability
  - `multiplier=0.3`: Gentle enhancement (recommended starting point)
  - `multiplier=1.0`: Full strength enhancement

- **Negative values (-1.0 to -0.1)**: Suppress the capability
  - `multiplier=-0.3`: Gentle suppression
  - `multiplier=-1.0`: Full strength suppression

- **Beyond ±1.0**: Can try larger values but may cause instability

## Best Practices

### Layer Selection

1. **Start with middle layers**: For most models, middle layers (around 40-60% depth) work best
   - 27B model (46 layers): Try layers 20-30
   - 9B model (32 layers): Try layers 15-20

2. **Use 3-5 layers**: Too few layers may be weak, too many may dilute the effect
   - Good: `--layers 22,24,26`
   - Okay: `--layers 20,22,24,26,28`
   - Avoid: `--layers 10,20,30,40` (too spread out)

3. **Experiment**: Different capabilities may respond better to different layers

### Memory Management

- **Large models (27B+)**: Use `--batch-size 1` or `2`
- **Medium models (7-13B)**: Use `--batch-size 4` or `8`
- **Small models (<7B)**: Use `--batch-size 8` or higher

If you get OOM (Out of Memory) errors:
1. Reduce `--batch-size`
2. Train on fewer layers
3. Use a smaller model

### Data Quality

- **Use all available samples** (default) for robust vectors
- Only use `--num-samples` for quick testing
- More data generally means more stable steering effects

## Troubleshooting

### "ModuleNotFoundError: No module named 'transformers'"

Install requirements:
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: Could not find contrast pairs file"

Make sure you're running from the project root, or specify `--data-dir`:
```bash
python experiments/train_vector.py \
    --contrast-type tom \
    --data-dir ./data/contrast_pairs/
```

### "CUDA out of memory"

Reduce batch size or train on fewer layers:
```bash
python experiments/train_vector.py \
    --contrast-type tom \
    --batch-size 1 \
    --layers 22,24,26
```

### Training is very slow

1. Reduce number of samples for testing: `--num-samples 50`
2. Train on fewer layers: `--layers 22,24,26` instead of `--layers all`
3. Increase batch size if you have memory: `--batch-size 4`
4. Use a smaller model: `--model google/gemma-2-9b-it`

## Next Steps

After training vectors:

1. **Test them**: Use `steering_interface.py` for interactive testing
2. **Evaluate**: Run systematic benchmarks on your use cases
3. **Experiment**: Try different multipliers and layer combinations
4. **Document**: Keep notes on which configurations work best for your tasks

## Reference

For more details on the research motivation and steering vector theory, see:
- `/CLAUDE.md` - Project overview and research goals
- `/docs/steering_vector_basics.md` - Technical reference
- `steering_interface.py` - Interactive testing interface
