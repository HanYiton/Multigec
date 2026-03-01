"""
Grammar correction reward functions for verl GRPO training.

Architecture:
    R_total = gate(R_format) × (w_preserve × R_preserve + w_correct × R_correct)

Sub-rewards:
    R_format   — Hard gate: non-empty, valid length ratio, no instruction leakage.
                 Fail → return -1.0 immediately.
    R_preserve — edit_distance_similarity(pred, source): how much original text is kept.
    R_correct  — chrF(pred, ref): character n-gram F-score for correction quality.
                 Language-agnostic, robust to morphological variation.

Scheduling:
    Sigmoid-based weight interpolation that shifts from preservation-heavy (early)
    to correction-heavy (late). Progress read from /dev/shm/multigec_progress.json.

Per-language normalization:
    Optional (off by default since GRPO already normalizes per-group).
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROGRESS_FILE = "/dev/shm/multigec_progress.json"
PROGRESS_REFRESH_INTERVAL = 100  # re-read file every N calls
ENABLE_LANG_NORMALIZATION = False  # off: GRPO already normalizes per-group
LANG_NORM_WARMUP = 30  # minimum samples before normalizing

FORMAT_FAIL_SCORE = -1.0
LENGTH_RATIO_BOUNDS = (0.3, 3.0)  # pred/source ratio must be within this range
CHRF_MAX_N = 6
CHRF_BETA = 1.0  # balanced precision/recall (avoids gaming recall by copying)


# ---------------------------------------------------------------------------
# Progress tracking (singleton cache)
# ---------------------------------------------------------------------------

_progress_cache = {"value": 0.0, "call_count": 0}


def _read_progress() -> float:
    """Read training progress from shared memory file. Returns p in [0, 1]."""
    cache = _progress_cache
    cache["call_count"] += 1

    if cache["call_count"] % PROGRESS_REFRESH_INTERVAL != 0:
        return cache["value"]

    try:
        with open(PROGRESS_FILE, "r") as f:
            data = json.load(f)
        step = data.get("step", 0)
        total = data.get("total_steps", 1)
        cache["value"] = min(1.0, max(0.0, step / max(total, 1)))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
        pass  # keep cached value (defaults to 0.0 = early-training weights)

    return cache["value"]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ---------------------------------------------------------------------------
# Per-language normalization (optional, Welford's online algorithm)
# ---------------------------------------------------------------------------

_lang_stats: dict = defaultdict(lambda: {"mean": 0.0, "var": 1.0, "count": 0})


def _update_lang_stats(lang: str, score: float) -> float:
    """Update running stats and return normalized score if enabled."""
    if not ENABLE_LANG_NORMALIZATION:
        return score

    stats = _lang_stats[lang]
    stats["count"] += 1
    n = stats["count"]

    delta = score - stats["mean"]
    stats["mean"] += delta / n
    delta2 = score - stats["mean"]
    stats["var"] += (delta * delta2 - stats["var"]) / n

    if n < LANG_NORM_WARMUP:
        return score

    std = max(math.sqrt(max(stats["var"], 0.0)), 1e-6)
    return (score - stats["mean"]) / std


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode for fair comparison."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# R_format: hard gate
# ---------------------------------------------------------------------------

_INSTRUCTION_LEAK_RE = re.compile(
    r"(?:please\s+)?correct\s+(?:all\s+)?(?:the\s+)?grammatical\s+errors?\s+"
    r"in\s+the\s+following",
    re.IGNORECASE,
)


def format_gate(pred: str, source: str) -> bool:
    """
    Hard format gate. Returns True if output is structurally acceptable.

    Checks:
      1. Non-empty after stripping whitespace
      2. Length ratio vs source within bounds (catches degenerate outputs)
      3. No instruction leakage (model echoed the prompt)
    """
    if not pred.strip():
        return False

    if source:
        ratio = len(pred) / max(len(source), 1)
        lo, hi = LENGTH_RATIO_BOUNDS
        if ratio < lo or ratio > hi:
            return False

    if _INSTRUCTION_LEAK_RE.search(pred):
        return False

    return True


# ---------------------------------------------------------------------------
# R_preserve: edit-distance similarity to source
# ---------------------------------------------------------------------------

def char_edit_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein distance (Wagner-Fischer, space-optimized)."""
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


def edit_distance_similarity(s1: str, s2: str) -> float:
    """Normalized similarity: 1 - (edit_distance / max_len). Returns [0, 1]."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - char_edit_distance(s1, s2) / max_len


# ---------------------------------------------------------------------------
# R_correct: chrF (character n-gram F-score)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> dict:
    """Extract character n-gram counts from text."""
    ngrams: dict = {}
    for i in range(len(text) - n + 1):
        ng = text[i:i + n]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


def chrf_score(
    pred: str,
    ref: str,
    max_n: int = CHRF_MAX_N,
    beta: float = CHRF_BETA,
) -> float:
    """
    Compute chrF score (character n-gram F-score).

    Character n-grams from 1 to max_n. Beta controls precision/recall balance:
      beta=1: balanced F1 (used here — avoids gaming recall by copying)
      beta=2: recall-weighted (standard chrF)
    """
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    total_p = 0.0
    total_r = 0.0
    count = 0

    for n in range(1, max_n + 1):
        pred_ng = _char_ngrams(pred, n)
        ref_ng = _char_ngrams(ref, n)

        if not pred_ng or not ref_ng:
            continue

        # Clipped matches (count each n-gram up to its reference frequency)
        matches = sum(min(pred_ng.get(ng, 0), c) for ng, c in ref_ng.items())

        pred_total = sum(pred_ng.values())
        ref_total = sum(ref_ng.values())

        total_p += matches / pred_total if pred_total else 0.0
        total_r += matches / ref_total if ref_total else 0.0
        count += 1

    if count == 0:
        return 0.0

    avg_p = total_p / count
    avg_r = total_r / count

    if avg_p + avg_r < 1e-12:
        return 0.0

    beta_sq = beta * beta
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


# ---------------------------------------------------------------------------
# Scheduling weights
# ---------------------------------------------------------------------------

def compute_weights(progress: float) -> tuple:
    """
    Sigmoid-smoothed weight schedule.

    Early training (p≈0): w_preserve ≈ 0.50, w_correct ≈ 0.50
    Late training  (p≈1): w_preserve ≈ 0.25, w_correct ≈ 0.75

    The shift is smooth (sigmoid) to avoid abrupt reward landscape changes.
    Weights always sum to 1.0.
    """
    s = _sigmoid(8.0 * (progress - 0.5))
    w_preserve = 0.5 - 0.25 * s
    w_correct = 0.5 + 0.25 * s
    return w_preserve, w_correct


# ---------------------------------------------------------------------------
# Main entry point (verl custom reward interface)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Compute reward score for grammar correction.

    Architecture:
        R = gate(R_format) × (w_preserve × R_preserve + w_correct × R_correct)

    Compatible with verl's custom reward function interface.

    Args:
        data_source: dataset identifier (e.g. "multigec_cs")
        solution_str: model-generated corrected text
        ground_truth: reference corrected text
        extra_info: dict with optional keys: "source" (original erroneous text),
                    "lang", "corpus"

    Returns:
        float: reward score. -1.0 for format failures, otherwise in [0, 1].
    """
    pred = normalize_text(solution_str)
    ref = normalize_text(ground_truth)

    source = ""
    if extra_info and "source" in extra_info:
        source = normalize_text(extra_info["source"])

    # --- Hard gate ---
    if not format_gate(pred, source):
        return FORMAT_FAIL_SCORE

    # --- Exact match shortcut ---
    if pred == ref:
        return 1.0

    # --- Sub-rewards ---
    r_preserve = edit_distance_similarity(pred, source) if source else 0.5
    r_correct = chrf_score(pred, ref)

    # --- Scheduling ---
    progress = _read_progress()
    w_preserve, w_correct = compute_weights(progress)

    # --- Weighted combination ---
    score = w_preserve * r_preserve + w_correct * r_correct

    # --- Optional per-language normalization ---
    lang = data_source.replace("multigec_", "", 1) if data_source.startswith("multigec_") else data_source
    score = _update_lang_stats(lang, score)

    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Progress writer utility (called by training loop or external script)
# ---------------------------------------------------------------------------

def write_progress(step: int, total_steps: int, path: str = PROGRESS_FILE) -> None:
    """
    Write training progress to shared memory file.

    Call this from a verl callback, training wrapper, or a cron-like watcher.
    The reward function reads this file to adjust scheduling weights.

    Args:
        step: current global training step
        total_steps: total expected steps
        path: file path (default: /dev/shm/multigec_progress.json)
    """
    import os
    data = {"step": step, "total_steps": total_steps}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic on POSIX
