# Evaluation Scripts

## eval_steering_vector.py

Evaluate a steering vector on a benchmark dataset.

### Usage

```bash
python scripts/eval_steering_vector.py \
    <model_name> \
    <intervention_path> \
    <steering_magnitude> \
    <completion_length> \
    <dataset_name>
```

### Parameters

- **model_name**: HuggingFace model identifier (e.g., `google/gemma-2-7b-it`)
- **intervention_path**: Path to steering vector pickle file (e.g., `./trained_vectors/gemma-2-7b-it/tom_all-layers_trained-on-92_20251030.pkl`)
- **steering_magnitude**: Multiplier for steering vector, typically between -1.0 and 1.0
- **completion_length**: Number of tokens to generate per example
- **dataset_name**: Name of JSON file in `./data/eval_data/` (e.g., `simpletom_behavior_qa.json`) or `all` to evaluate all datasets

### Examples

**Single dataset:**
```bash
python scripts/eval_steering_vector.py \
    google/gemma-3-12b-it \
    ./trained_vectors/gemma-3-12b-it/tom_all-layers_trained-on-92_20251030.pkl \
    0.5 \
    150 \
    simpletom_behavior_qa.json
```

**All datasets:**
```bash
python scripts/eval_steering_vector.py \
    google/gemma-3-12b-it \
    ./trained_vectors/gemma-3-12b-it/tom_all-layers_trained-on-92_20251030.pkl \
    0.5 \
    150 \
    all
```

### Output

Results are saved to:
```
./data/results/{model_name}/{steering_vector}_{magnitude}/{dataset_name}.json
```

For example:
```
./data/results/gemma-3-12b-it/tom_all-layers_trained-on-92_20251030_0.5/simpletom_behavior_qa.json
```

Each result includes the original dataset fields plus a `generated_completion` field with the model's output.
