#!/usr/bin/env python3
"""
Generate Theory of Mind contrast pairs.

High ToM = Model understands false beliefs and mental states
Low ToM = Model only states reality, doesn't track perspectives

Based on tasks from the ToM literature:
1. Unexpected Contents: Container labeled S2, actually contains S1
2. Unexpected Transfer: Object moved from A to B while protagonist away
3. Irony/Sarcasm: Statement contradicts reality
"""

import json
from pathlib import Path
from generate_examples import tsks1, tsks2, tsks3

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_FILE = PROJECT_DIR / "tom_pairs.json"


def create_unexpected_contents_pairs():
    """Create pairs from unexpected contents tasks (false belief about container contents)."""
    pairs = []

    for task in tsks1:
        txt = task['txt']
        o1 = task['o1']  # What's actually inside (reality)
        o2 = task['o2']  # What the label says (false belief)
        c = task['c']    # Container
        xnam = task['xnam']
        xpro = task['xpro']

        # Replace placeholders
        scenario = (txt.replace('CX', c)
                       .replace('S1', o1)
                       .replace('S2', o2)
                       .replace('XNAM', xnam)
                       .replace('XPRO', xpro)
                       .replace('POS_PRO', task['pos_pro']))

        # Create belief question (what protagonist thinks based on label)
        belief_prompt = f"{scenario} {xnam.capitalize()} tells a friend about the {c}."

        pairs.append({
            "scenario": belief_prompt,
            "high_tom": f"{xnam} says the {c} contains {o2}.",
            "low_tom": f"{xnam} says the {c} contains {o1}.",
            "category": "unexpected_contents",
            "explanation": f"High ToM understands {xnam} believes it's {o2} (based on label). Low ToM just states reality ({o1})."
        })

        # Also create a variant with direct question
        question_prompt = f"{scenario} A friend asks {xnam} what is in the {c}."

        pairs.append({
            "scenario": question_prompt,
            "high_tom": f"{xnam} answers '{o2}'.",
            "low_tom": f"{xnam} answers '{o1}'.",
            "category": "unexpected_contents",
            "explanation": f"High ToM: {xnam} would say '{o2}' (false belief from label). Low ToM: just reality '{o1}'."
        })

    return pairs


def create_unexpected_transfer_pairs():
    """Create pairs from unexpected transfer tasks (false belief about object location)."""
    pairs = []

    for task in tsks2:
        txt = task['txt']
        o1 = task['o1']  # Current location (reality)
        o2 = task['o2']  # Original location (false belief)
        q2_stem = task.get('q2', '')

        # Extract protagonist name from the story
        # Most stories mention protagonist by name
        import re
        names = re.findall(r'\b([A-Z][a-z]+)\b', txt[:200])
        protagonist = names[-1] if names else "the protagonist"

        # Create completion prompts
        search_prompt = f"{txt}"

        pairs.append({
            "scenario": search_prompt,
            "high_tom": f"{protagonist} looks in the {o2}.",
            "low_tom": f"{protagonist} looks in the {o1}.",
            "category": "unexpected_transfer",
            "explanation": f"High ToM: protagonist believes it's in {o2} (original location). Low ToM: just reality ({o1})."
        })

    return pairs


def create_irony_sarcasm_pairs():
    """Create pairs from irony/sarcasm tasks (literal vs non-literal interpretation)."""
    pairs = []

    for task in tsks3:
        txt = task['txt']
        txt_reverse = task['txt_reverse']

        # Extract the statement and context
        if 'mused' in txt or 'said' in txt or 'exclaimed' in txt or 'commented' in txt or 'laughed' in txt or 'giggled' in txt:
            # Find the quote
            import re
            quote_match = re.search(r"'([^']+)'", txt)
            context_before_quote = txt[:quote_match.start()] if quote_match else txt

            # Determine if it's sarcastic (txt) or literal (txt_reverse)
            # In txt, context contradicts the statement (sarcasm)
            # In txt_reverse, context supports the statement (literal)

            question = txt.split('Question:')[1].strip()

            pairs.append({
                "scenario": txt,
                "high_tom": "No, they were being sarcastic.",
                "low_tom": "Yes, they meant it literally.",
                "category": "irony_sarcasm",
                "explanation": "High ToM understands sarcasm (context contradicts statement). Low ToM takes it literally."
            })

            pairs.append({
                "scenario": txt_reverse,
                "high_tom": "Yes, they meant it sincerely.",
                "low_tom": "No, they were being sarcastic.",
                "category": "irony_sarcasm",
                "explanation": "High ToM understands sincerity (context supports statement). Low ToM misreads as sarcasm."
            })

    return pairs


def create_simple_tom_pairs():
    """Create simple, clear-cut ToM contrast pairs."""

    simple_pairs = [
        {
            "scenario": "Here is a bag filled with chocolate. There is no popcorn in this bag. Yet, the label on this bag says 'popcorn' and not 'chocolate'. Sam finds the bag. Sam has never seen this bag before. Sam calls a friend.",
            "high_tom": "Sam tells their friend 'I found a bag of popcorn.'",
            "low_tom": "Sam tells their friend 'I found a bag of chocolate.'",
            "category": "simple_false_belief",
            "explanation": "High ToM: Sam believes it's popcorn (label). Low ToM: just reality (chocolate)."
        },
        {
            "scenario": "Alice puts her keys in the drawer. She leaves for work. While Alice is away, her roommate Bob moves the keys to the kitchen counter. Alice comes home.",
            "high_tom": "Alice looks for her keys in the drawer.",
            "low_tom": "Alice looks for her keys on the kitchen counter.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Alice believes keys are in drawer. Low ToM: just states where keys actually are."
        },
        {
            "scenario": "A box is labeled 'TOOLS' but actually contains toys. Jenny has never opened this box before. She needs to find something.",
            "high_tom": "Jenny opens the box expecting to find tools.",
            "low_tom": "Jenny opens the box expecting to find toys.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Jenny's expectation based on label. Low ToM: what's actually inside."
        },
        {
            "scenario": "Tom thinks his meeting is at 3pm, but it was rescheduled to 2pm. Tom didn't receive the notification. It's currently 1:55pm.",
            "high_tom": "Tom is relaxed, thinking he has plenty of time.",
            "low_tom": "Tom is rushing, knowing the meeting starts soon.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Tom's false belief (3pm). Low ToM: actual reality (2pm)."
        },
        {
            "scenario": "Sara sees John putting cookies in the blue jar. After John leaves, Sara moves the cookies to the red jar. John returns and wants a cookie.",
            "high_tom": "John reaches for the blue jar.",
            "low_tom": "John reaches for the red jar.",
            "category": "simple_false_belief",
            "explanation": "High ToM: John's outdated belief (blue jar). Low ToM: current reality (red jar)."
        },
        {
            "scenario": "The hotel room has a mini-fridge labeled 'Empty - Out of Service'. However, it's actually stocked with drinks. A guest checks in for the first time.",
            "high_tom": "The guest doesn't bother checking the mini-fridge.",
            "low_tom": "The guest opens the mini-fridge and finds drinks.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Guest's belief based on sign. Low ToM: what's actually true."
        },
        {
            "scenario": "Emma believes that Madrid is the capital of Portugal. Her friend asks her: 'What's the capital of Portugal?'",
            "high_tom": "Emma answers 'Madrid'.",
            "low_tom": "Emma answers 'Lisbon'.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Emma's false belief (Madrid). Low ToM: actual fact (Lisbon)."
        },
        {
            "scenario": "A child watches their parent hide a present in the closet. The parent doesn't know the child saw this. The parent asks the child to guess where the present is.",
            "high_tom": "The child pretends not to know and guesses wrong.",
            "low_tom": "The child immediately says 'in the closet'.",
            "category": "simple_false_belief",
            "explanation": "High ToM: Child understands parent's false belief (that child doesn't know). Low ToM: just states reality."
        },
    ]

    return simple_pairs


def main():
    """Generate all ToM contrast pairs."""

    print("🧠 Generating Theory of Mind contrast pairs...")
    print(f"📁 Output: {OUTPUT_FILE}")
    print()

    all_pairs = []

    # Generate simple pairs first
    print("Generating simple ToM pairs...")
    simple_pairs = create_simple_tom_pairs()
    all_pairs.extend(simple_pairs)
    print(f"  ✅ {len(simple_pairs)} simple pairs")

    # Generate from unexpected contents tasks
    print("Generating unexpected contents pairs...")
    contents_pairs = create_unexpected_contents_pairs()
    all_pairs.extend(contents_pairs)
    print(f"  ✅ {len(contents_pairs)} unexpected contents pairs")

    # Generate from unexpected transfer tasks
    print("Generating unexpected transfer pairs...")
    transfer_pairs = create_unexpected_transfer_pairs()
    all_pairs.extend(transfer_pairs)
    print(f"  ✅ {len(transfer_pairs)} unexpected transfer pairs")

    # Generate from irony/sarcasm tasks
    print("Generating irony/sarcasm pairs...")
    irony_pairs = create_irony_sarcasm_pairs()
    all_pairs.extend(irony_pairs)
    print(f"  ✅ {len(irony_pairs)} irony/sarcasm pairs")

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_pairs, f, indent=2)

    print()
    print("=" * 70)
    print(f"✅ Generated {len(all_pairs)} ToM contrast pairs!")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print()

    # Show category breakdown
    from collections import Counter
    categories = Counter(p['category'] for p in all_pairs)
    print("Pairs by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    print()
    print("Sample pairs:")
    for i, pair in enumerate(all_pairs[:3]):
        print(f"\n[{i+1}] {pair['category']}")
        print(f"Scenario: {pair['scenario'][:80]}...")
        print(f"High ToM: {pair['high_tom']}")
        print(f"Low ToM:  {pair['low_tom']}")
        print(f"Why: {pair['explanation']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
