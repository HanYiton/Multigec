"""
将增强后的 JSONL 数据转换为 verl 所需的 Parquet 格式。

verl GRPO 所需的 Parquet 格式：
  - data_source: str  — 数据来源标识
  - prompt: list[dict] — chat messages (仅包含 user turn)
  - ability: str       — 能力标签
  - reward_model: dict — 奖励函数所需的信息
  - extra_info: dict   — 额外信息

verl SFT 所需的格式：
  - 标准 HuggingFace chat 格式 (messages 列表)
"""

import json
import argparse
import os
from typing import List, Dict


def jsonl_to_sft_parquet(jsonl_path: str, output_path: str, max_samples: int = None):
    """将 JSONL 转为 SFT 训练用的 Parquet。"""
    import pandas as pd

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                sample = json.loads(line)
                records.append({
                    "messages": sample["messages"],
                    "lang": sample["lang"],
                    "id": sample["id"],
                })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"SFT Parquet: {len(records)} samples -> {output_path}")


def jsonl_to_grpo_parquet(jsonl_path: str, output_path: str, max_samples: int = None):
    """将 JSONL 转为 GRPO 训练用的 Parquet。"""
    import pandas as pd

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if line.strip():
                sample = json.loads(line)
                records.append({
                    "data_source": f"multigec_{sample['lang']}",
                    "prompt": [{"role": "user", "content": sample["prompt"]}],
                    "ability": "grammar_correction",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": sample["target"],
                    },
                    "extra_info": {
                        "id": sample["id"],
                        "lang": sample["lang"],
                        "corpus": sample["corpus"],
                        "source": sample["source"],
                    },
                })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"GRPO Parquet: {len(records)} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL")
    parser.add_argument("--output_sft", type=str, default=None, help="Output SFT parquet")
    parser.add_argument("--output_grpo", type=str, default=None, help="Output GRPO parquet")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_sft or args.output_grpo), exist_ok=True)

    if args.output_sft:
        jsonl_to_sft_parquet(args.input, args.output_sft, args.max_samples)
    if args.output_grpo:
        jsonl_to_grpo_parquet(args.input, args.output_grpo, args.max_samples)


if __name__ == "__main__":
    main()
