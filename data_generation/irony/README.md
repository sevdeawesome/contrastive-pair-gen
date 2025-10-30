# Irony Contrast Pairs Dataset

This directory contains all the scripts and data used to generate the irony contrast pairs dataset for steering vector training.

## Dataset Source

The dataset was created using the [Ironic Corpus](https://www.kaggle.com/datasets/rtatman/ironic-corpus) from Kaggle, which contains 1,949 Reddit comments labeled as ironic (537) or non-ironic (1,412).

## Generation Process

### 1. Download Dataset
- Used `kagglehub` to download the ironic corpus
- Script: `download_dataset.py`

### 2. Explore Dataset
- Analyzed the structure and content of the dataset
- Identified labeled ironic vs non-ironic examples
- Script: `explore_dataset.py`

### 3. Generate Contrast Pairs
- Selected 100 ironic examples from the corpus
- Used OpenAI API (gpt-4o-mini) to generate literal/non-ironic versions
- Applied quality validation:
  - Minimal structural difference
  - Similar length (within 2x ratio)
  - Same core meaning, different tone
- Script: `generate_pairs.py`
- Output: `irony_pairs_raw.json`

### 4. Clean and Finalize
- Cleaned whitespace and formatting
- Applied additional quality filters
- Generated final dataset with 97 high-quality pairs
- Script: `create_final_dataset.py`
- Output: `../irony_pairs.json` (main directory)

## Dataset Format

The final dataset (`irony_pairs.json`) follows this structure:

```json
[
  {
    "ironic": "Ironic statement using sarcasm or opposite meaning",
    "literal": "Literal/sincere version of the same statement"
  },
  ...
]
```

## Dataset Statistics

- **Total pairs:** 97
- **Average ironic length:** 154.7 characters
- **Average literal length:** 150.1 characters
- **Average length difference:** 22.6 characters

## Quality Principles

Each pair follows steering vector best practices:

1. **Minimal difference** - Only the tone/ironic delivery changes
2. **Same length** - Within reasonable bounds (~2x ratio max)
3. **Same structure** - Similar sentence patterns maintained
4. **Clear target** - Isolates ironic vs literal expression
5. **Complete** - Full sentences, not truncated
6. **Consistent format** - All pairs follow same template

## Example Pairs

**Pair 1:**
- **Ironic:** "Keep attacking the one world leader who stands up to fags, guys. Whoever said conservatives are their own worst enemies?"
- **Literal:** "Keep supporting the one world leader who stands up for traditional values, everyone. Whoever said conservatives are their own best allies?"

**Pair 2:**
- **Ironic:** "$780 in food stamps (plus WIC)! That's some lavish dining!"
- **Literal:** "$780 in food stamps (plus WIC)! That's a helpful amount for dining!"

**Pair 3:**
- **Ironic:** "Imagine if schools spent $70 million on education."
- **Literal:** "Imagine if schools invested $70 million in education."

## Files in this Directory

- `download_dataset.py` - Downloads the Kaggle ironic corpus
- `explore_dataset.py` - Analyzes dataset structure
- `generate_pairs.py` - Generates contrast pairs using OpenAI API
- `create_final_dataset.py` - Cleans and finalizes the dataset
- `irony_pairs_raw.json` - Raw generated pairs before cleaning
- `README.md` - This file

## Usage

To regenerate the dataset:

1. Ensure you have a `.env` file with `OPENAI_API_KEY`
2. Install dependencies: `pip install kagglehub openai python-dotenv pandas`
3. Run the generation pipeline:
   ```bash
   python download_dataset.py
   python explore_dataset.py
   python generate_pairs.py
   python create_final_dataset.py
   ```

## Notes

- The OpenAI API was used to generate non-ironic versions because creating literal counterparts requires understanding nuanced sarcasm and tone
- The validation ensures pairs meet quality standards for steering vector training
- Some pairs were rejected during generation if they didn't meet quality criteria
