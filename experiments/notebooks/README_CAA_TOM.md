# CAA Theory of Mind Notebook

## Overview

This is a verbatim adaptation of `caa_sycophancy.ipynb` that uses the Theory of Mind (ToM) dataset instead of the sycophancy dataset.

## Files Created

1. **`caa_tom.ipynb`** - Main notebook (verbatim copy with ToM data)
2. **`tom_data_loader.py`** - Data loader utility to convert ToM format to CAA format
3. **`test_tom_loader.py`** - Verification script for data loader

## Key Differences from Original

### Dataset Path Configuration
- **GLOBAL VARIABLE** at top of notebook:
  ```python
  DATASET_PATH = "../../data/contrast_pairs/theory_of_mind_multiple_choice.json"
  ```

### Data Loading (Cell ~11-12)
- Uses `tom_data_loader.py` instead of loading JSON files directly
- Converts ToM format to CAA-compatible format:
  - `prompt` → `question`
  - `high_tom_answer` → `answer_matching_behavior` (what we steer TOWARD)
  - `low_tom_answer` → `answer_not_matching_behavior` (what we steer AWAY FROM)

### Updated Comments
- Changed "sycophantic answer" to "high ToM answer" in evaluation description

## Data Format Conversion

### Original ToM Format
```json
{
  "prompt": "Question text with choices...",
  "high_tom_answer": "A",
  "low_tom_answer": "B",
  "category": "unexpected_contents",
  "explanation": "..."
}
```

### CAA-Compatible Format (after conversion)
```json
{
  "question": "Question text with choices...",
  "answer_matching_behavior": "A",
  "answer_not_matching_behavior": "B"
}
```

## Dataset Statistics
- **Total examples**: 92
- **Train split**: 73 examples (80%)
- **Test split**: 19 examples (20%)
- **Split method**: Random with seed=42 (reproducible)

## Usage

Simply open and run `caa_tom.ipynb` in Jupyter. All data loading is handled automatically through the global `DATASET_PATH` variable.

## Verification

Run the test script to verify data loading:
```bash
cd experiments/notebooks
python test_tom_loader.py
```

Expected output:
- Total examples: 92
- Train size: 73
- Test size: 19
- All format checks passed ✓
