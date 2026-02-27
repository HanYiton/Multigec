"""
Grammar correction reward functions for verl GRPO training.

Reward signals:
1. Exact match: full score if output matches reference exactly
2. Character-level edit distance ratio: partial credit based on similarity
3. Length penalty: penalize outputs that are too short or too long
4. Copy penalty: penalize outputs identical to the (erroneous) input
"""

import re
import unicodedata
from typing import Optional


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode for fair comparison."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_edit_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein distance (Wagner-Fischer)."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s1) + 1))
    for j in range(1, len(s2) + 1):
        curr = [j] + [0] * len(s1)
        for i in range(1, len(s1) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[-1]


def edit_distance_similarity(pred: str, ref: str) -> float:
    """Normalized similarity: 1 - (edit_distance / max_len). Returns [0, 1]."""
    if not pred and not ref:
        return 1.0
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0
    dist = char_edit_distance(pred, ref)
    return 1.0 - dist / max_len


def length_ratio_penalty(pred: str, ref: str, tolerance: float = 0.3) -> float:
    """
    Penalize if predicted length deviates too much from reference length.
    Returns 1.0 if within tolerance, decays linearly otherwise.
    """
    if not ref:
        return 1.0
    ratio = len(pred) / max(len(ref), 1)
    if 1.0 - tolerance <= ratio <= 1.0 + tolerance:
        return 1.0
    deviation = abs(ratio - 1.0) - tolerance
    return max(0.0, 1.0 - deviation)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Compute reward score for grammar correction.

    Compatible with verl's custom reward function interface.

    Args:
        data_source: dataset identifier (e.g. "multigec_cs")
        solution_str: model-generated corrected text
        ground_truth: reference corrected text
        extra_info: dict with optional keys: "source" (original erroneous text),
                    "lang", "corpus"

    Returns:
        float: reward score in [0, 1]
    """
    pred = normalize_text(solution_str)
    ref = normalize_text(ground_truth)

    # Empty output → 0
    if not pred:
        return 0.0

    # Exact match → full score
    if pred == ref:
        return 1.0

    # Base similarity from edit distance
    sim = edit_distance_similarity(pred, ref)

    # Length penalty
    len_pen = length_ratio_penalty(pred, ref)

    # Copy penalty: if the model just copies the erroneous input, reduce reward
    copy_penalty = 1.0
    if extra_info and "source" in extra_info:
        source = normalize_text(extra_info["source"])
        if source and pred == source and pred != ref:
            # Model copied the input unchanged when it should have corrected
            copy_penalty = 0.3

    score = sim * len_pen * copy_penalty

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
