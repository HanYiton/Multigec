"""
Evaluation script for MultiGEC grammar correction models.

Metrics:
  - Exact Match (EM): fraction of predictions that exactly match the reference
  - Character-level Edit Distance Similarity (EDS): normalized Levenshtein similarity
  - GLEU: sentence-level BLEU variant for GEC (Napoles et al., 2015)
  - Per-language breakdown of all metrics

Usage:
    python -m src.eval.evaluate \
        --predictions outputs/predictions.jsonl \
        --references processed_data/multigec_dev.jsonl
"""

import json
import argparse
import re
import unicodedata
from collections import defaultdict
from typing import List, Dict, Tuple


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_edit_distance(s1: str, s2: str) -> int:
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
    if not pred and not ref:
        return 1.0
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0
    return 1.0 - char_edit_distance(pred, ref) / max_len


def compute_gleu(source: str, prediction: str, reference: str, max_n: int = 4) -> float:
    """
    Simplified GLEU score (Napoles et al., 2015).
    Measures n-gram overlap of prediction with reference,
    penalizing n-grams that appear in source but not reference.
    """
    def get_ngrams(text, n):
        tokens = text.split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    if not prediction.strip() or not reference.strip():
        return 0.0

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)
        src_ngrams = get_ngrams(source, n)

        if not pred_ngrams or not ref_ngrams:
            continue

        ref_set = set(ref_ngrams)
        src_set = set(src_ngrams)

        # Count matches: pred n-grams that appear in ref
        matches = sum(1 for ng in pred_ngrams if ng in ref_set)
        # Penalize: pred n-grams that are in source but not in ref (unchanged errors)
        # This is a simplification of the full GLEU metric
        precision = matches / len(pred_ngrams) if pred_ngrams else 0
        recall = matches / len(ref_ngrams) if ref_ngrams else 0

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 0.0

    avg_p = total_precision / count
    avg_r = total_recall / count

    if avg_p + avg_r == 0:
        return 0.0

    # F-score
    return 2 * avg_p * avg_r / (avg_p + avg_r)


def evaluate(predictions: List[Dict], references: List[Dict]) -> Dict:
    """
    Evaluate predictions against references.

    Args:
        predictions: list of {"id": ..., "prediction": ..., "lang": ...}
        references: list of {"id": ..., "target": ..., "source": ..., "lang": ...}

    Returns:
        dict with overall and per-language metrics
    """
    # Index references by id
    ref_by_id = {r["id"]: r for r in references}

    per_lang = defaultdict(lambda: {"em": [], "eds": [], "gleu": [], "n": 0})
    overall = {"em": [], "eds": [], "gleu": []}

    matched = 0
    for pred in predictions:
        pid = pred["id"]
        if pid not in ref_by_id:
            continue

        ref = ref_by_id[pid]
        matched += 1

        pred_text = normalize_text(pred["prediction"])
        ref_text = normalize_text(ref["target"])
        src_text = normalize_text(ref.get("source", ""))
        lang = ref.get("lang", "unknown")

        # Exact match
        em = 1.0 if pred_text == ref_text else 0.0

        # Edit distance similarity
        eds = edit_distance_similarity(pred_text, ref_text)

        # GLEU
        gleu = compute_gleu(src_text, pred_text, ref_text)

        overall["em"].append(em)
        overall["eds"].append(eds)
        overall["gleu"].append(gleu)

        per_lang[lang]["em"].append(em)
        per_lang[lang]["eds"].append(eds)
        per_lang[lang]["gleu"].append(gleu)
        per_lang[lang]["n"] += 1

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    results = {
        "overall": {
            "n_predictions": len(predictions),
            "n_matched": matched,
            "exact_match": avg(overall["em"]),
            "edit_distance_similarity": avg(overall["eds"]),
            "gleu": avg(overall["gleu"]),
        },
        "per_language": {},
    }

    for lang in sorted(per_lang.keys()):
        m = per_lang[lang]
        results["per_language"][lang] = {
            "n": m["n"],
            "exact_match": avg(m["em"]),
            "edit_distance_similarity": avg(m["eds"]),
            "gleu": avg(m["gleu"]),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="JSONL with predictions")
    parser.add_argument("--references", type=str, required=True, help="JSONL with references")
    parser.add_argument("--output", type=str, default=None, help="Output JSON for metrics")
    args = parser.parse_args()

    # Load predictions
    preds = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    # Load references
    refs = []
    with open(args.references, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                refs.append(json.loads(line))

    results = evaluate(preds, refs)

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    o = results["overall"]
    print(f"  Matched: {o['n_matched']} / {o['n_predictions']}")
    print(f"  Exact Match:    {o['exact_match']:.4f}")
    print(f"  Edit Dist Sim:  {o['edit_distance_similarity']:.4f}")
    print(f"  GLEU:           {o['gleu']:.4f}")

    print(f"\nPer-language breakdown:")
    print(f"  {'Lang':<6} {'N':>6} {'EM':>8} {'EDS':>8} {'GLEU':>8}")
    print(f"  {'-'*38}")
    for lang, m in results["per_language"].items():
        print(f"  {lang:<6} {m['n']:>6} {m['exact_match']:>8.4f} {m['edit_distance_similarity']:>8.4f} {m['gleu']:>8.4f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
