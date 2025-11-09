"""
Data loader for Theory of Mind dataset.
Converts ToM data format to CAA-compatible format.
"""

import json
from typing import List, Dict


def load_tom_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """
    Load Theory of Mind dataset and convert to CAA format.

    Args:
        dataset_path: Path to the ToM dataset JSON file

    Returns:
        List of dictionaries with keys:
        - question: The prompt with choices
        - answer_matching_behavior: High ToM answer (what we steer toward)
        - answer_not_matching_behavior: Low ToM answer (what we steer away from)
    """
    with open(dataset_path, 'r') as f:
        tom_data = json.load(f)

    caa_format_data = []

    for item in tom_data:
        caa_item = {
            "question": item["prompt"],
            "answer_matching_behavior": item["high_tom_answer"],
            "answer_not_matching_behavior": item["low_tom_answer"]
        }
        caa_format_data.append(caa_item)

    return caa_format_data


def create_train_test_split(data: List[Dict[str, str]],
                            train_ratio: float = 0.8) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into train and test sets.

    Args:
        data: Full dataset
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of (train_data, test_data)
    """
    import random

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data
