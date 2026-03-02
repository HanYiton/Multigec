"""
数据诊断与对比工具：分析增强前后各语种的关键统计量。

指标：
  - 样本数量、恒等样本数/占比、增强样本数/占比
  - 平均 source/target 长度（字符 & 词）
  - 平均编辑距离、编辑相似度、错误率
  - 每样本平均修改数（编辑操作数 / source 长度）

用法：
  # 单文件分析
  python -m src.data.analyze_data --input data.jsonl

  # 增强前后对比
  python -m src.data.analyze_data --before raw.jsonl --after augmented.jsonl

  # 输出 JSON
  python -m src.data.analyze_data --before raw.jsonl --after augmented.jsonl --output stats.json
"""

import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text utilities (duplicated from rewards to keep this module standalone)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(s1: str, s2: str) -> int:
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


# ---------------------------------------------------------------------------
# Per-language statistics
# ---------------------------------------------------------------------------

def compute_lang_stats(samples: List[Dict]) -> Dict[str, Dict]:
    """
    Compute per-language statistics for a list of samples.

    Each sample should have: id, lang, source, target.

    Returns dict keyed by language code, plus an "overall" key.
    Each value contains:
      - n_samples: total count
      - n_identity: source == target (no correction needed)
      - n_augmented: id ends with _aug
      - n_identity_injected: id ends with _id
      - identity_ratio: n_identity / n_samples
      - avg_source_len_char: mean character length of source
      - avg_target_len_char: mean character length of target
      - avg_source_len_word: mean word count of source
      - avg_target_len_word: mean word count of target
      - avg_edit_distance: mean char edit distance (source → target)
      - avg_edit_similarity: mean 1 - edit_dist / max_len
      - error_rate: 1 - avg_edit_similarity
      - avg_correction_density: mean edit_dist / source_len
    """
    accum: Dict[str, Dict] = defaultdict(lambda: {
        "n_samples": 0,
        "n_identity": 0,
        "n_augmented": 0,
        "n_identity_injected": 0,
        "source_len_char": [],
        "target_len_char": [],
        "source_len_word": [],
        "target_len_word": [],
        "edit_distances": [],
        "edit_similarities": [],
        "correction_densities": [],
    })

    for s in samples:
        lang = s.get("lang", "unknown")
        src = _normalize(s.get("source", ""))
        tgt = _normalize(s.get("target", ""))
        sid = s.get("id", "")

        for key in (lang, "__overall__"):
            a = accum[key]
            a["n_samples"] += 1

            if src == tgt:
                a["n_identity"] += 1
            if sid.endswith("_aug"):
                a["n_augmented"] += 1
            if sid.endswith("_id"):
                a["n_identity_injected"] += 1

            a["source_len_char"].append(len(src))
            a["target_len_char"].append(len(tgt))
            a["source_len_word"].append(len(src.split()))
            a["target_len_word"].append(len(tgt.split()))

            if src and tgt:
                ed = _edit_distance(src, tgt)
                max_len = max(len(src), len(tgt))
                sim = 1.0 - ed / max_len if max_len > 0 else 1.0
                density = ed / len(src) if len(src) > 0 else 0.0
            elif not src and not tgt:
                ed, sim, density = 0, 1.0, 0.0
            else:
                max_len = max(len(src), len(tgt))
                ed = max_len
                sim = 0.0
                density = 1.0

            a["edit_distances"].append(ed)
            a["edit_similarities"].append(sim)
            a["correction_densities"].append(density)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    result = {}
    for key, a in accum.items():
        n = a["n_samples"]
        result[key] = {
            "n_samples": n,
            "n_identity": a["n_identity"],
            "n_augmented": a["n_augmented"],
            "n_identity_injected": a["n_identity_injected"],
            "identity_ratio": a["n_identity"] / n if n else 0.0,
            "avg_source_len_char": _avg(a["source_len_char"]),
            "avg_target_len_char": _avg(a["target_len_char"]),
            "avg_source_len_word": _avg(a["source_len_word"]),
            "avg_target_len_word": _avg(a["target_len_word"]),
            "avg_edit_distance": _avg(a["edit_distances"]),
            "avg_edit_similarity": _avg(a["edit_similarities"]),
            "error_rate": 1.0 - _avg(a["edit_similarities"]),
            "avg_correction_density": _avg(a["correction_densities"]),
        }

    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_stats(
    before: Dict[str, Dict],
    after: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Compare two stat dicts (before/after augmentation).

    Returns a dict with the same language keys. Each value has:
      - before: {metrics}
      - after: {metrics}
      - delta: {metric_name: after - before} for numeric fields
    """
    all_langs = sorted(set(before.keys()) | set(after.keys()))
    result = {}

    numeric_keys = [
        "n_samples", "n_identity", "n_augmented", "n_identity_injected",
        "identity_ratio", "avg_source_len_char", "avg_target_len_char",
        "avg_source_len_word", "avg_target_len_word",
        "avg_edit_distance", "avg_edit_similarity", "error_rate",
        "avg_correction_density",
    ]

    empty = {k: 0 for k in numeric_keys}

    for lang in all_langs:
        b = before.get(lang, empty)
        a = after.get(lang, empty)
        delta = {}
        for k in numeric_keys:
            bv = b.get(k, 0)
            av = a.get(k, 0)
            delta[k] = av - bv
        result[lang] = {"before": b, "after": a, "delta": delta}

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_single_stats(stats: Dict[str, Dict]) -> str:
    """Format stats for a single dataset as a readable table."""
    lines = []
    overall = stats.get("__overall__", {})
    lang_keys = sorted(k for k in stats if k != "__overall__")

    lines.append(f"{'Lang':<6} {'N':>6} {'Ident':>6} {'Ident%':>7} {'Aug':>5} "
                 f"{'SrcLen':>7} {'TgtLen':>7} {'SrcWrd':>7} {'TgtWrd':>7} "
                 f"{'EditDist':>9} {'EditSim':>8} {'ErrRate':>8} {'CorrDen':>8}")
    lines.append("-" * 110)

    def _row(label, s):
        return (f"{label:<6} {s['n_samples']:>6} {s['n_identity']:>6} "
                f"{s['identity_ratio']:>7.1%} {s['n_augmented']:>5} "
                f"{s['avg_source_len_char']:>7.1f} {s['avg_target_len_char']:>7.1f} "
                f"{s['avg_source_len_word']:>7.1f} {s['avg_target_len_word']:>7.1f} "
                f"{s['avg_edit_distance']:>9.2f} {s['avg_edit_similarity']:>8.4f} "
                f"{s['error_rate']:>8.4f} {s['avg_correction_density']:>8.4f}")

    for lang in lang_keys:
        lines.append(_row(lang, stats[lang]))

    lines.append("-" * 110)
    if overall:
        lines.append(_row("ALL", overall))

    return "\n".join(lines)


def format_comparison(comparison: Dict[str, Dict]) -> str:
    """Format a before/after comparison as a readable table."""
    lines = []
    overall_key = "__overall__"
    lang_keys = sorted(k for k in comparison if k != overall_key)

    # Header
    lines.append(f"{'Lang':<6} │ {'N (b→a)':>13} {'Ident% (b→a)':>17} "
                 f"{'ErrRate (b→a)':>19} {'CorrDen (b→a)':>19} "
                 f"{'Aug#':>5} {'IdInj#':>6}")
    lines.append("─" * 100)

    def _delta_str(val):
        if val > 0:
            return f"+{val:.4f}"
        elif val < 0:
            return f"{val:.4f}"
        return "  0"

    def _row(label, c):
        b, a, d = c["before"], c["after"], c["delta"]
        n_str = f"{b.get('n_samples',0):>5}→{a.get('n_samples',0):<5}"
        id_str = f"{b.get('identity_ratio',0):>6.1%}→{a.get('identity_ratio',0):<6.1%}"
        er_str = f"{b.get('error_rate',0):>7.4f}→{a.get('error_rate',0):<7.4f}"
        cd_str = f"{b.get('avg_correction_density',0):>7.4f}→{a.get('avg_correction_density',0):<7.4f}"
        aug = a.get("n_augmented", 0)
        idinj = a.get("n_identity_injected", 0)
        return f"{label:<6} │ {n_str:>13} {id_str:>17} {er_str:>19} {cd_str:>19} {aug:>5} {idinj:>6}"

    for lang in lang_keys:
        lines.append(_row(lang, comparison[lang]))

    lines.append("─" * 100)
    if overall_key in comparison:
        lines.append(_row("ALL", comparison[overall_key]))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyze per-language data characteristics before/after augmentation.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single JSONL file to analyze")
    group.add_argument("--before", type=str, help="JSONL file before augmentation")
    parser.add_argument("--after", type=str, help="JSONL file after augmentation (requires --before)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for stats")
    args = parser.parse_args()

    if args.input:
        # Single file analysis
        samples = load_jsonl(args.input)
        stats = compute_lang_stats(samples)
        print(f"\nDataset: {args.input} ({len(samples)} samples)\n")
        print(format_single_stats(stats))

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"\nStats saved to {args.output}")

    elif args.before:
        if not args.after:
            parser.error("--after is required when using --before")

        before_samples = load_jsonl(args.before)
        after_samples = load_jsonl(args.after)
        before_stats = compute_lang_stats(before_samples)
        after_stats = compute_lang_stats(after_samples)
        comparison = compare_stats(before_stats, after_stats)

        print(f"\nBefore: {args.before} ({len(before_samples)} samples)")
        print(f"After:  {args.after} ({len(after_samples)} samples)\n")
        print(format_comparison(comparison))

        # Also print detailed before/after tables
        print(f"\n{'='*60}")
        print("BEFORE augmentation:")
        print(f"{'='*60}")
        print(format_single_stats(before_stats))

        print(f"\n{'='*60}")
        print("AFTER augmentation:")
        print(f"{'='*60}")
        print(format_single_stats(after_stats))

        if args.output:
            out = {
                "before": before_stats,
                "after": after_stats,
                "comparison": comparison,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()
