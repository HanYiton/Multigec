"""
解析 MultiGEC 2025 数据集的 Markdown 格式文件，
将 (原文, 纠正文) 对提取并转换为统一的 JSON Lines / Parquet 格式。

支持：
  - 句子对齐 (sentence_aligned=True) 语料：逐行对齐
  - 篇章对齐 (sentence_aligned=False) 语料：按 essay_id 对齐段落
  - 多参考 (ref1, ref2, …)：每条参考生成一条独立样本
"""

import re
import os
import json
import yaml
import glob
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ── 语言代码到全名映射 ─────────────────────────────────────────────
LANG_MAP = {
    "cs": "Czech",   "en": "English", "et": "Estonian", "de": "German",
    "el": "Greek",   "is": "Icelandic", "it": "Italian", "lv": "Latvian",
    "ru": "Russian", "sl": "Slovene",  "sv": "Swedish", "uk": "Ukrainian",
}

# 语言全名到代码
LANG_NAME_TO_CODE = {
    "czech": "cs", "english": "en", "estonian": "et", "german": "de",
    "greek": "el", "icelandic": "is", "italian": "it", "latvian": "lv",
    "russian": "ru", "slovene": "sl", "swedish": "sv", "ukrainian": "uk",
}


def parse_md_file(filepath: str) -> Dict[str, str]:
    """解析 MultiGEC markdown 文件，返回 {essay_id: text} 字典。"""
    essays = {}
    current_id = None
    current_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^###\s+essay_id\s*=\s*(.+)$", line)
            if m:
                if current_id is not None:
                    essays[current_id] = "\n".join(current_lines).strip()
                current_id = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_id is not None:
            essays[current_id] = "\n".join(current_lines).strip()

    return essays


def discover_corpora(data_dir: str) -> List[Dict]:
    """自动发现 data/ 下所有语料及其元数据。"""
    corpora = []
    metadata_files = glob.glob(os.path.join(data_dir, "**/metadata.yaml"), recursive=True)

    for meta_path in sorted(metadata_files):
        corpus_dir = os.path.dirname(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        lang_code = meta.get("target_language", "")
        corpus_name = os.path.basename(corpus_dir)
        sentence_aligned = meta.get("sentence_aligned", False)

        # 找到所有 ref 键
        ref_keys = sorted([k for k in meta.keys() if k.startswith("reference_essays_")])
        n_refs = len(ref_keys)

        corpora.append({
            "lang_code": lang_code,
            "lang_name": LANG_MAP.get(lang_code, lang_code),
            "corpus_name": corpus_name,
            "corpus_dir": corpus_dir,
            "sentence_aligned": sentence_aligned,
            "n_refs": n_refs,
            "metadata": meta,
        })

    return corpora


def find_split_files(corpus_dir: str, lang_code: str, corpus_name: str):
    """找到某语料的 train/dev/test 原文和参考文件。"""
    md_files = glob.glob(os.path.join(corpus_dir, "*.md"))
    splits = {}

    for md_file in md_files:
        basename = os.path.basename(md_file)
        # 跳过 README/CHANGELOG
        if basename.lower() in ("readme.md", "changelog.md"):
            continue

        # 尝试匹配 split 和 type (orig/ref1/ref2...)
        for split in ("train", "dev", "test"):
            if f"-{split}" in basename or f"_{split}" in basename:
                if "-orig-" in basename or "_orig_" in basename:
                    splits.setdefault(split, {})["orig"] = md_file
                else:
                    ref_match = re.search(r"[-_](ref\d+)[-_]", basename)
                    if ref_match:
                        ref_key = ref_match.group(1)
                        splits.setdefault(split, {}).setdefault("refs", {})[ref_key] = md_file

    return splits


def align_essays_to_samples(
    orig_essays: Dict[str, str],
    ref_essays: Dict[str, str],
    sentence_aligned: bool,
) -> List[Tuple[str, str]]:
    """
    将原文和参考文对齐为 (source, target) 样本列表。
    - sentence_aligned=True: 按行拆分对齐
    - sentence_aligned=False: 保持整篇对齐
    """
    samples = []
    common_ids = set(orig_essays.keys()) & set(ref_essays.keys())

    for eid in sorted(common_ids):
        orig_text = orig_essays[eid]
        ref_text = ref_essays[eid]

        if not orig_text.strip() or not ref_text.strip():
            continue

        if sentence_aligned:
            orig_lines = [l for l in orig_text.split("\n") if l.strip()]
            ref_lines = [l for l in ref_text.split("\n") if l.strip()]
            # 逐行对齐
            for o, r in zip(orig_lines, ref_lines):
                if o.strip():
                    samples.append((o.strip(), r.strip()))
        else:
            # 整篇作为一个样本
            samples.append((orig_text.strip(), ref_text.strip()))

    return samples


def build_instruction(source: str, lang_code: str) -> str:
    """为每条样本构造 instruction-following 格式的 prompt。"""
    lang_name = LANG_MAP.get(lang_code, lang_code)
    return (
    f"Please identify whether the {lang_name} sentence I provide contains any grammatical errors. "
    "If there are grammatical errors, please correct them with the minimal necessary changes. "
    "If there are no grammatical errors, please reply with the original sentence. "
    "You must output your final answer in the following format: "
    "<answer> your corrected sentence, or the original sentence </answer> "
    f"The sentence you need to correct is: {source}"
    )


def build_sft_sample(source: str, target: str, lang_code: str, corpus: str, split: str, ref_key: str) -> Dict:
    """构造 SFT 训练样本 (chat 格式)。"""
    instruction = build_instruction(source, lang_code)
    uid = hashlib.md5(f"{lang_code}:{corpus}:{source[:200]}:{ref_key}".encode()).hexdigest()[:12]

    return {
        "id": uid,
        "lang": lang_code,
        "corpus": corpus,
        "split": split,
        "ref": ref_key,
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": target},
        ],
        # 以下字段用于 GRPO (prompt-only 格式)
        "prompt": instruction,
        "source": source,
        "target": target,
    }


def process_all(data_dir: str, output_dir: str, max_source_len: int = 1024, max_target_len: int = 1024):
    """处理所有语料，输出 JSONL + Parquet。"""
    os.makedirs(output_dir, exist_ok=True)
    corpora = discover_corpora(data_dir)

    all_samples = {"train": [], "dev": [], "test": []}
    stats = {}

    for corpus_info in corpora:
        lang_code = corpus_info["lang_code"]
        corpus_name = corpus_info["corpus_name"]
        corpus_dir = corpus_info["corpus_dir"]
        sentence_aligned = corpus_info["sentence_aligned"]

        print(f"\n{'='*60}")
        print(f"Processing: {lang_code} / {corpus_name}")
        print(f"  Directory: {corpus_dir}")
        print(f"  Sentence aligned: {sentence_aligned}")

        split_files = find_split_files(corpus_dir, lang_code, corpus_name)
        corpus_stats = {}

        for split in ("train", "dev", "test"):
            if split not in split_files:
                print(f"  [{split}] No files found, skipping.")
                continue

            sf = split_files[split]
            if "orig" not in sf:
                print(f"  [{split}] No orig file found, skipping.")
                continue

            orig_essays = parse_md_file(sf["orig"])
            refs = sf.get("refs", {})

            if not refs:
                print(f"  [{split}] No ref files found, skipping.")
                continue

            split_count = 0
            for ref_key, ref_path in sorted(refs.items()):
                ref_essays = parse_md_file(ref_path)
                pairs = align_essays_to_samples(orig_essays, ref_essays, sentence_aligned)

                for source, target in pairs:
                    # 粗略长度过滤 (按字符数)
                    if len(source) > max_source_len * 4 or len(target) > max_target_len * 4:
                        continue
                    if not source or not target:
                        continue

                    sample = build_sft_sample(source, target, lang_code, corpus_name, split, ref_key)
                    all_samples[split].append(sample)
                    split_count += 1

            corpus_stats[split] = split_count
            print(f"  [{split}] Generated {split_count} samples")

        stats[f"{lang_code}/{corpus_name}"] = corpus_stats

    # ── 输出 ──────────────────────────────────────────────────────
    for split, samples in all_samples.items():
        if not samples:
            continue

        # JSONL
        jsonl_path = os.path.join(output_dir, f"multigec_{split}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n[{split}] Wrote {len(samples)} samples to {jsonl_path}")

    # Parquet (用于 verl GRPO)
    try:
        import pandas as pd
        for split, samples in all_samples.items():
            if not samples:
                continue
            # GRPO parquet 格式：需要 prompt 列 (list of dicts)
            grpo_records = []
            for s in samples:
                grpo_records.append({
                    "data_source": f"multigec_{s['lang']}_{s['corpus']}",
                    "prompt": [{"role": "user", "content": s["prompt"]}],
                    "ability": "grammar_correction",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": s["target"],
                    },
                    "extra_info": {
                        "id": s["id"],
                        "lang": s["lang"],
                        "corpus": s["corpus"],
                        "source": s["source"],
                        "split": split,
                    },
                })
            df = pd.DataFrame(grpo_records)
            parquet_path = os.path.join(output_dir, f"multigec_{split}.parquet")
            df.to_parquet(parquet_path, index=False)
            print(f"[{split}] Wrote Parquet to {parquet_path}")
    except ImportError:
        print("\n[WARNING] pandas not installed, skipping Parquet output. Install with: pip install pandas pyarrow")

    # 统计报告
    print(f"\n{'='*60}")
    print("STATISTICS SUMMARY")
    print(f"{'='*60}")
    total = {s: 0 for s in ("train", "dev", "test")}
    for corpus_key, corpus_stats in sorted(stats.items()):
        for split, count in corpus_stats.items():
            total[split] += count
        print(f"  {corpus_key}: {corpus_stats}")
    print(f"\n  TOTAL: train={total['train']}, dev={total['dev']}, test={total['test']}")

    return all_samples, stats


def main():
    parser = argparse.ArgumentParser(description="Parse MultiGEC 2025 data into training format")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to MultiGEC data directory")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    parser.add_argument("--max_source_len", type=int, default=1024, help="Max source char length (x4 for filtering)")
    parser.add_argument("--max_target_len", type=int, default=1024, help="Max target char length (x4 for filtering)")
    args = parser.parse_args()

    process_all(args.data_dir, args.output_dir, args.max_source_len, args.max_target_len)


if __name__ == "__main__":
    main()
