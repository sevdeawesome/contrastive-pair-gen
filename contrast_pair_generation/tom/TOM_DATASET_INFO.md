# Theory of Mind (ToM) Contrast Pairs Dataset

Generated: 2025-10-19

## Overview

This dataset contains 92 high-quality Theory of Mind contrast pairs designed to train steering vectors that control the degree to which a language model demonstrates theory of mind reasoning.

## What is Theory of Mind?

Theory of Mind is the ability to attribute mental states (beliefs, desires, intentions, knowledge) to oneself and others, and to understand that others have beliefs and knowledge that differ from one's own.

## Dataset Purpose

Train steering vectors to control how well models:
- Track false beliefs (what someone *thinks* vs. what's *actually true*)
- Understand different perspectives and mental states
- Distinguish between reality and beliefs
- Recognize sarcasm/irony (literal vs. non-literal meaning)

## Contrast Pair Structure

Each pair consists of:
- **scenario**: A situation involving false beliefs or perspective-taking
- **high_tom**: Completion demonstrating understanding of mental states and false beliefs
- **low_tom**: Completion that only states reality, ignoring perspectives
- **category**: Type of ToM task
- **explanation**: Why high_tom shows ToM and low_tom doesn't

## Dataset Statistics

- **Total pairs**: 92
- **Categories**: 4 distinct ToM task types

### Category Distribution

| Category | Count | Description |
|----------|-------|-------------|
| unexpected_contents | 40 | Container labeled S2, actually contains S1 |
| irony_sarcasm | 24 | Statement contradicts reality (literal vs. sarcastic) |
| unexpected_transfer | 20 | Object moved while protagonist away |
| simple_false_belief | 8 | Clear-cut false belief scenarios |

## Task Types Explained

### 1. Unexpected Contents (40 pairs)
**Classic false belief task**: Container has misleading label

**Example:**
```json
{
  "scenario": "Here is a bag filled with chocolate. There is no popcorn in this bag. Yet, the label on this bag says 'popcorn' and not 'chocolate'. Sam finds the bag. Sam has never seen this bag before. Sam calls a friend.",
  "high_tom": "Sam tells their friend 'I found a bag of popcorn.'",
  "low_tom": "Sam tells their friend 'I found a bag of chocolate.'",
  "explanation": "High ToM: Sam believes it's popcorn (label). Low ToM: just reality (chocolate)."
}
```

**Key insight**: High ToM understands that Sam will believe the label (false belief), while low ToM just states what's actually in the bag.

### 2. Unexpected Transfer (20 pairs)
**Classic location false belief**: Object moved while protagonist absent

**Example:**
```json
{
  "scenario": "Alice puts her keys in the drawer. She leaves for work. While Alice is away, her roommate Bob moves the keys to the kitchen counter. Alice comes home.",
  "high_tom": "Alice looks for her keys in the drawer.",
  "low_tom": "Alice looks for her keys on the kitchen counter.",
  "explanation": "High ToM: Alice believes keys are in drawer. Low ToM: just states where keys actually are."
}
```

**Key insight**: High ToM tracks that Alice's knowledge is outdated (she thinks keys are still in drawer), while low ToM just states current reality.

### 3. Irony/Sarcasm (24 pairs)
**Non-literal meaning**: Statement contradicts context

**Example:**
```json
{
  "scenario": "It was Valentine's Day and Jamie had received no cards. 'It appears the girls really fancy you', laughed his flatmate. Question: Did the flatmate think the girls really fancy Jamie?",
  "high_tom": "No, they were being sarcastic.",
  "low_tom": "Yes, they meant it literally.",
  "explanation": "High ToM understands sarcasm (context contradicts statement). Low ToM takes it literally."
}
```

**Key insight**: High ToM recognizes when context contradicts the literal meaning (sarcasm), while low ToM interprets everything literally.

### 4. Simple False Belief (8 pairs)
**Clear-cut scenarios** for easy evaluation

**Example:**
```json
{
  "scenario": "Tom thinks his meeting is at 3pm, but it was rescheduled to 2pm. Tom didn't receive the notification. It's currently 1:55pm.",
  "high_tom": "Tom is relaxed, thinking he has plenty of time.",
  "low_tom": "Tom is rushing, knowing the meeting starts soon.",
  "explanation": "High ToM: Tom's false belief (3pm). Low ToM: actual reality (2pm)."
}
```

**Key insight**: High ToM tracks Tom's false belief about the meeting time, while low ToM states the actual time.

## Quality Criteria

✅ **Clear contrast**: High ToM vs. Low ToM differ precisely on perspective-taking
✅ **Same length**: Responses are similar length (within ~5-15 tokens)
✅ **Minimal difference**: Only the mental state reasoning changes
✅ **Complete sentences**: Natural, fluent language
✅ **Varied scenarios**: 4 different task types from ToM literature
✅ **Explanations included**: Each pair has explanation of the difference

## Based on Research

This dataset is based on established Theory of Mind tasks from cognitive science literature:

1. **Unexpected Contents Task** (Gopnik & Astington, 1988)
   - Classic "Smarties task" / "deceptive box" paradigm
   - Tests understanding of false beliefs about container contents

2. **Unexpected Transfer Task** (Baron-Cohen et al., 1985)
   - Classic "Sally-Anne task" paradigm
   - Tests understanding of false beliefs about object locations

3. **Irony/Sarcasm Detection** (Happé, 1993)
   - Tests understanding of non-literal meaning
   - Requires tracking speaker's beliefs vs. literal statement

## Usage Examples

### Training Steering Vectors

```python
# High ToM direction: Model tracks mental states, false beliefs
# Low ToM direction: Model only states reality

# Use these pairs to extract activation differences
# Then steer model along ToM dimension
```

### Evaluation

```python
# Test if model can:
# 1. Distinguish between belief and reality
# 2. Track what protagonist knows vs. what's actually true
# 3. Recognize sarcasm/irony
# 4. Adopt different perspectives
```

## Sample Pairs from Each Category

### Unexpected Contents
```
Scenario: "A box labeled 'TOOLS' contains toys. Jenny has never opened it."
High ToM: "Jenny expects to find tools."
Low ToM: "Jenny expects to find toys."
```

### Unexpected Transfer
```
Scenario: "John puts cookies in blue jar. Sara moves them to red jar while John is away. John returns."
High ToM: "John reaches for the blue jar."
Low ToM: "John reaches for the red jar."
```

### Irony/Sarcasm
```
Scenario: "Empty function hall. 'Clearly people were keen on coming to your party', giggled Sylvia."
High ToM: "No, she was being sarcastic."
Low ToM: "Yes, she meant it literally."
```

### Simple False Belief
```
Scenario: "Emma believes Madrid is the capital of Portugal. Friend asks: 'What's the capital?'"
High ToM: "Emma answers 'Madrid'."
Low ToM: "Emma answers 'Lisbon'."
```

## Key Insight

The fundamental distinction:
- **High ToM**: Tracks what people *believe* (even if it's wrong)
- **Low ToM**: Only tracks what's *actually true* (ignoring beliefs)

This is crucial for:
- Social understanding
- Communication
- Empathy
- Predicting behavior
- Understanding deception and mistakes

## Files

- `generate_tom_pairs.py` - Generation script
- `generate_examples.py` - Source task templates
- `context.md` - Background on ToM tasks
- `../tom_pairs.json` - Final dataset (92 pairs)

## Citation

Based on classic Theory of Mind tasks:
- Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a "theory of mind"?
- Gopnik, A., & Astington, J. W. (1988). Children's understanding of representational change
- Happé, F. G. (1993). Communicative competence and theory of mind in autism
