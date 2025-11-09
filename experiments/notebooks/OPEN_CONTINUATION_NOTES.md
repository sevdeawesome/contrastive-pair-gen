# Open Continuation Steering Vectors - Implementation Notes

## Overview

This document explains the key differences between the multiple-choice CAA approach and the open continuation approach implemented in `steering_vectors_open_continuation.ipynb`.

## Key Changes from Multiple-Choice Approach

### 1. Data Format

**Multiple Choice (Original CAA):**
- Training pairs: `(prompt + answer_A, prompt + answer_B)`
- Extracts activations at the position where the model decides between A/B
- Example: `"...(A) Too much time\n(B) Too little time"`

**Open Continuation (This Notebook):**
- Training pairs: `(positive_text, negative_text)`
- Extracts activations from the full contrastive examples
- Example: `"I can help with that"` vs `"The assistant can help with that"`

### 2. Token Position for Reading Activations

**Multiple Choice:**
- Used `-2` (second-to-last token) to capture the A/B decision point

**Open Continuation:**
- Uses `-1` (last token) to capture the full context
- **EPISTEMIC: Moderately Confident** - Last token should contain full sequence context, but could experiment with:
  - Average pooling across all tokens
  - Last N tokens average
  - Specific positions if there's a clear "decision point"

### 3. Training Data Construction

The notebook supports multiple data formats via the `PREFIX_KEY` parameter:

**Simple Pairs (self_other.json):**
```python
PREFIX_KEY = None
POSITIVE_KEY = 'self_subject'
NEGATIVE_KEY = 'other_subject'
# Uses the full text from each key
```

**Prefix + Completion (theory_of_mind.json):**
```python
PREFIX_KEY = 'scenario'
POSITIVE_KEY = 'high_tom'
NEGATIVE_KEY = 'low_tom'
# Concatenates: scenario + high_tom vs scenario + low_tom
```

**EPISTEMIC: Uncertain** - For data with scenarios/contexts, it's unclear whether we should:
- Include the full scenario + completion (current approach)
- Use only the completion/response portion
- Use the scenario as the test prompt and completions as training targets

This likely depends on what aspect you want to steer (the understanding vs the response).

### 4. Steering Application

**Multiple Choice:**
- Applied during evaluation to measure probability shift between choices
- Steering affects logits at decision point

**Open Continuation:**
- Applied during generation to influence token sampling
- Steering affects all token positions (via `min_token_index=0`)
- **EPISTEMIC: Moderately Confident** - Applying to all tokens works, but alternatives to consider:
  - Apply only to generated tokens (not prompt tokens)
  - Apply with increasing/decreasing strength over generation
  - Apply only to specific token positions

### 5. Evaluation Method

**Multiple Choice:**
- Quantitative: Measure probability of choosing correct answer
- Clear success metric (higher probability = better steering)

**Open Continuation:**
- Qualitative: Compare generated text across steering multipliers
- Requires human judgment of whether outputs show desired properties
- **EPISTEMIC: Certain** - This is inherently more subjective, but more flexible for open-ended tasks

## Hyperparameters and Uncertainties

### Layer Selection

```python
STEERING_LAYERS = [12, 13, 14]  # Middle layers for 9B model
```

**EPISTEMIC: Uncertain** - Optimal layers likely vary by:
- Model size (2B vs 9B vs 27B)
- Task type (ToM vs self-reference vs harmfulness)
- What you're trying to steer (shallow features vs deep reasoning)

**Suggestions to try:**
- Early layers (1-5): Surface features, syntax
- Middle layers (10-15): Semantic content, concepts
- Late layers (20+): Task-specific reasoning, output formatting
- All layers: Maximum effect but less interpretable

### Read Token Position

```python
READ_TOKEN_INDEX = -1  # Last token
```

**EPISTEMIC: Moderately Confident** - Last token probably captures full context, but worth testing:
- `-1`: Full context (current)
- `-2` to `-5`: Earlier positions
- Average of last N tokens

### Steering Multipliers

```python
MULTIPLIERS = [-2.0, 0.0, 2.0]
```

**EPISTEMIC: Certain** - Interpretation:
- **Positive values**: Steer toward `POSITIVE_KEY` examples
- **Negative values**: Steer toward `NEGATIVE_KEY` examples
- **Zero**: No steering (baseline)
- **Magnitude**: Strength of steering (typical range: -3.0 to +3.0)

Larger magnitudes = stronger steering but may reduce coherence.

### Min Token Index for Application

```python
steering_vector.apply(model, multiplier=mult, min_token_index=0)
```

**EPISTEMIC: Uncertain** - Currently applies to all tokens. Alternatives:
- `min_token_index=len(prompt_tokens)`: Only affect generated tokens
- `min_token_index=10`: Skip first few tokens

May need to experiment based on what you're steering.

## Dataset-Specific Considerations

### Self-Other Distinction
- **Goal**: Make model use "I/my/me" vs "they/their/them" language
- **Expected effect**: Very clear and easy to observe
- **Suggested test prompts**: Any task description or capability question

### Theory of Mind
- **Goal**: Make model reason about beliefs vs just state facts
- **Expected effect**: More nuanced, may need careful prompt design
- **Suggested test prompts**: False belief scenarios, knowledge attribution

### Harmfulness
- **Goal**: Make model more/less likely to refuse or comply with requests
- **Expected effect**: Should be very obvious (refusal language vs compliance)
- **Suggested test prompts**: Borderline requests, potentially sensitive topics

## Common Issues and Debugging

### Steering Has No Effect
1. **Check layer selection**: Try different layers or all layers (`STEERING_LAYERS = None`)
2. **Increase multiplier magnitude**: Try ±5.0 or ±10.0
3. **Verify data loading**: Print example pairs to ensure contrast is clear
4. **Check token position**: Try different `READ_TOKEN_INDEX` values

### Steering Reduces Coherence
1. **Lower multiplier magnitude**: Try ±0.5 or ±1.0
2. **Use fewer layers**: Focus on specific layer range
3. **Adjust generation parameters**: Lower temperature, adjust top_p

### Steering Works on Training Distribution But Not Test Cases
1. **Training data may be too specific**: Add more diverse examples
2. **Test prompts may be out of distribution**: Make them more similar to training
3. **May need more training examples**: Increase dataset size

## Next Steps and Experiments

1. **Layer ablation study**: Train vectors on different layer ranges, compare effects
2. **Token position study**: Try different read positions, compare quality
3. **Magnitude sweep**: Test multipliers from -5.0 to +5.0 to find optimal range
4. **Cross-dataset transfer**: Train on one dataset, test on another
5. **Multi-vector composition**: Combine multiple steering vectors (e.g., ToM + Self-reference)

## Code Verification

**I cannot test this notebook without GPU access.** Before running on your data, verify:
- [ ] Paths in config section are correct for your setup
- [ ] Data files are in expected format (JSON with specified keys)
- [ ] Model name is available and you have access/memory
- [ ] Dependencies are installed (`steering-vectors` package)

## References

- Original CAA paper: https://arxiv.org/abs/2312.06681
- Steering vectors library: https://github.com/steering-vectors/steering-vectors
- Your steering vector docs: `steering_vector_docs.md`
