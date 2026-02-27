"""
数据增强策略：针对多语种语法纠错任务。

策略：
1. 同义参考混合 (Multi-Reference Mixing)
   — 同一个原文，不同参考纠正均作为独立样本（已在 parse 阶段完成）
2. 恒等映射注入 (Identity Injection)
   — 将无错误的参考文同时作为输入和输出，教模型"不要过度纠正"
3. 回译噪声 (Back-Translation Noise)
   — 对已纠正文添加可控噪声，制造新的训练对
4. 语言标签均衡采样 (Balanced Sampling)
   — 对低资源语言过采样，对高资源语言下采样
"""

import json
import random
import copy
from collections import Counter, defaultdict
from typing import List, Dict, Optional


def inject_identity_samples(samples: List[Dict], ratio: float = 0.1, seed: int = 42) -> List[Dict]:
    """
    从现有样本中随机选取一部分，将 target 同时作为 source 和 target，
    构造"无错误则原样输出"样本。
    ratio: 添加的恒等样本占原样本的比例
    """
    rng = random.Random(seed)
    n_identity = int(len(samples) * ratio)
    selected = rng.sample(samples, min(n_identity, len(samples)))

    identity_samples = []
    for s in selected:
        new_sample = copy.deepcopy(s)
        target = new_sample["target"]
        new_sample["source"] = target
        new_sample["target"] = target
        # 更新 messages
        new_sample["messages"][0]["content"] = new_sample["messages"][0]["content"].rsplit("\n\n", 1)[0] + "\n\n" + target
        new_sample["messages"][1]["content"] = target
        new_sample["prompt"] = new_sample["messages"][0]["content"]
        new_sample["id"] = new_sample["id"] + "_id"
        identity_samples.append(new_sample)

    return identity_samples


def add_synthetic_noise(text: str, lang: str, noise_rate: float = 0.1, seed: int = None) -> str:
    """
    给一段文本注入简单的合成噪声：
    - 删除随机空格
    - 交换相邻字符
    - 随机大小写替换
    - 删除随机标点
    """
    rng = random.Random(seed)
    chars = list(text)
    n_ops = max(1, int(len(chars) * noise_rate))

    for _ in range(n_ops):
        op = rng.choice(["swap", "delete_space", "case", "delete_punct"])
        if len(chars) < 3:
            break

        if op == "swap":
            idx = rng.randint(0, len(chars) - 2)
            if chars[idx].isalpha() and chars[idx + 1].isalpha():
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        elif op == "delete_space":
            space_indices = [i for i, c in enumerate(chars) if c == " "]
            if space_indices:
                idx = rng.choice(space_indices)
                chars.pop(idx)

        elif op == "case":
            alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
            if alpha_indices:
                idx = rng.choice(alpha_indices)
                chars[idx] = chars[idx].swapcase()

        elif op == "delete_punct":
            punct_indices = [i for i, c in enumerate(chars) if c in ".,;:!?"]
            if punct_indices:
                idx = rng.choice(punct_indices)
                chars.pop(idx)

    return "".join(chars)


def generate_noisy_samples(samples: List[Dict], ratio: float = 0.2, noise_rate: float = 0.05, seed: int = 42) -> List[Dict]:
    """
    从参考文本出发，添加合成噪声生成新的训练对。
    """
    rng = random.Random(seed)
    n_noisy = int(len(samples) * ratio)
    selected = rng.sample(samples, min(n_noisy, len(samples)))

    noisy_samples = []
    for i, s in enumerate(selected):
        new_sample = copy.deepcopy(s)
        target = new_sample["target"]
        noisy_source = add_synthetic_noise(target, new_sample["lang"], noise_rate, seed=seed + i)
        new_sample["source"] = noisy_source
        new_sample["target"] = target
        new_sample["messages"][0]["content"] = new_sample["messages"][0]["content"].rsplit("\n\n", 1)[0] + "\n\n" + noisy_source
        new_sample["messages"][1]["content"] = target
        new_sample["prompt"] = new_sample["messages"][0]["content"]
        new_sample["id"] = new_sample["id"] + "_aug"
        noisy_samples.append(new_sample)

    return noisy_samples


def balanced_sampling(
    samples: List[Dict],
    target_per_lang: Optional[int] = None,
    strategy: str = "sqrt",
    seed: int = 42,
) -> List[Dict]:
    """
    按语言进行均衡采样。

    strategy:
      - "uniform": 每种语言采样相同数量
      - "sqrt": 每种语言按 sqrt(原始数量) 比例采样
      - "log": 每种语言按 log(原始数量) 比例采样
    """
    import math
    rng = random.Random(seed)

    # 按语言分组
    by_lang = defaultdict(list)
    for s in samples:
        by_lang[s["lang"]].append(s)

    lang_counts = {lang: len(samps) for lang, samps in by_lang.items()}
    max_count = max(lang_counts.values())

    if target_per_lang is not None:
        targets = {lang: target_per_lang for lang in lang_counts}
    elif strategy == "uniform":
        avg = sum(lang_counts.values()) // len(lang_counts)
        targets = {lang: avg for lang in lang_counts}
    elif strategy == "sqrt":
        # 按 sqrt 比例计算目标数量，然后归一化到总样本数
        total = sum(lang_counts.values())
        sqrt_sum = sum(math.sqrt(c) for c in lang_counts.values())
        targets = {lang: int(total * math.sqrt(c) / sqrt_sum) for lang, c in lang_counts.items()}
    elif strategy == "log":
        total = sum(lang_counts.values())
        log_sum = sum(math.log(c + 1) for c in lang_counts.values())
        targets = {lang: int(total * math.log(c + 1) / log_sum) for lang, c in lang_counts.items()}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    result = []
    for lang, target in targets.items():
        lang_samples = by_lang[lang]
        if target >= len(lang_samples):
            # 过采样
            result.extend(lang_samples)
            extra = target - len(lang_samples)
            result.extend(rng.choices(lang_samples, k=extra))
        else:
            # 下采样
            result.extend(rng.sample(lang_samples, target))

    rng.shuffle(result)
    return result


def augment_dataset(
    jsonl_path: str,
    output_path: str,
    identity_ratio: float = 0.1,
    noise_ratio: float = 0.2,
    noise_rate: float = 0.05,
    balance_strategy: str = "sqrt",
    seed: int = 42,
):
    """完整的数据增强管线。"""
    # 加载
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Original samples: {len(samples)}")
    lang_dist = Counter(s["lang"] for s in samples)
    print(f"Language distribution: {dict(sorted(lang_dist.items()))}")

    # 1. 恒等注入
    id_samples = inject_identity_samples(samples, ratio=identity_ratio, seed=seed)
    print(f"Identity samples added: {len(id_samples)}")

    # 2. 噪声增强
    noisy_samples = generate_noisy_samples(samples, ratio=noise_ratio, noise_rate=noise_rate, seed=seed)
    print(f"Noisy augmented samples added: {len(noisy_samples)}")

    # 3. 合并
    all_samples = samples + id_samples + noisy_samples
    print(f"Total before balancing: {len(all_samples)}")

    # 4. 均衡采样
    balanced = balanced_sampling(all_samples, strategy=balance_strategy, seed=seed)
    print(f"Total after balancing ({balance_strategy}): {len(balanced)}")

    lang_dist_after = Counter(s["lang"] for s in balanced)
    print(f"Language distribution after balancing: {dict(sorted(lang_dist_after.items()))}")

    # 输出
    with open(output_path, "w", encoding="utf-8") as f:
        for s in balanced:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote augmented data to {output_path}")
    return balanced


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--identity_ratio", type=float, default=0.1)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--noise_rate", type=float, default=0.05)
    parser.add_argument("--balance_strategy", type=str, default="sqrt", choices=["uniform", "sqrt", "log"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.identity_ratio, args.noise_ratio, args.noise_rate, args.balance_strategy, args.seed)
