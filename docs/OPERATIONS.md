# MultiGEC: Multilingual Grammar Error Correction

Complete operational guide for training grammar correction models on the
MultiGEC 2025 dataset using SFT + GRPO reinforcement learning with verl.

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                    Training Pipeline                    │
│                                                        │
│  Raw Data (12 langs)                                   │
│       │                                                │
│       ▼                                                │
│  parse_multigec.py  ─→  JSONL (train/dev/test)         │
│       │                                                │
│       ▼                                                │
│  augmentation.py    ─→  Augmented JSONL                 │
│       │                                                │
│       ▼                                                │
│  prepare_verl_data.py ─→ Parquet (SFT + GRPO)          │
│       │                                                │
│       ├──────────────────┐                             │
│       ▼                  ▼                             │
│  SFT Training       GRPO Training                      │
│  (DeepSpeed ZeRO-2)  (verl framework)                  │
│       │                  │                             │
│       ▼                  ▼                             │
│  inference.py  ─→  evaluate.py  ─→  Metrics            │
└────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **Minimum**: 2x NVIDIA A30 (24GB each) or equivalent
- **Recommended**: 2x NVIDIA A100 (40GB+) for larger models
- **Storage**: ~50GB for data, models, and checkpoints
- **RAM**: 64GB+

## Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <repo-url> && cd Multigec
git submodule update --init --recursive

# Create environment
conda create -n multigec python=3.10 -y
conda activate multigec

# Install dependencies
pip install -r requirements.txt

# Install verl
pip install verl
# OR from source:
# cd third_party/verl && pip install -e . && cd ../..

# Install flash-attention (recommended)
pip install flash-attn --no-build-isolation
```

### 2. Run Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

Or run each step individually:

```bash
# Step 1: Parse data
python -m src.data.parse_multigec --data_dir data --output_dir processed_data

# Step 2: Augment
python -m src.data.augmentation \
    --input processed_data/multigec_train.jsonl \
    --output processed_data/multigec_train_aug.jsonl

# Step 3: Prepare parquet
python -m src.data.prepare_verl_data \
    --input processed_data/multigec_train_aug.jsonl \
    --output_sft processed_data/multigec_train_sft.parquet \
    --output_grpo processed_data/multigec_train_grpo.parquet

# Step 4: SFT
deepspeed --num_gpus=2 scripts/train_sft.py --config configs/sft_config.yaml

# Step 5: GRPO
bash scripts/train_grpo.sh outputs/sft/final \
    processed_data/multigec_train_grpo.parquet \
    processed_data/multigec_dev_grpo.parquet

# Step 6: Evaluate
python scripts/inference.py \
    --model_path outputs/sft/final \
    --input processed_data/multigec_dev.jsonl \
    --output outputs/predictions_dev.jsonl

python -m src.eval.evaluate \
    --predictions outputs/predictions_dev.jsonl \
    --references processed_data/multigec_dev.jsonl \
    --output outputs/metrics_dev.json
```

## Data Processing Details

### Supported Languages (12)

| Code | Language   | Corpora                          | Alignment   |
|------|-----------|----------------------------------|-------------|
| cs   | Czech     | NatForm, NatWebInf, Romani, SecLearn | Sentence  |
| en   | English   | WriteAndImprove2024              | Sentence    |
| et   | Estonian   | EIC, EKIL2                       | Mixed       |
| de   | German    | Merlin                           | Sentence    |
| el   | Greek     | GLCII                            | Document    |
| is   | Icelandic | IceEC, IceL2EC                   | Mixed       |
| it   | Italian   | Merlin                           | Sentence    |
| lv   | Latvian   | LaVA                             | Document    |
| ru   | Russian   | rulec-gec                        | Sentence    |
| sl   | Slovene   | Solar-Eval                       | Document    |
| sv   | Swedish   | SweLL_gold                       | Document    |
| uk   | Ukrainian | ua-gec                           | Sentence    |

### Data Format

Each sample in JSONL format:
```json
{
  "id": "a1b2c3d4e5f6",
  "lang": "cs",
  "corpus": "NatForm",
  "split": "train",
  "ref": "ref1",
  "messages": [
    {"role": "user", "content": "Please correct all grammatical errors in the following Czech text. Output only the corrected text without any explanation.\n\n<erroneous text>"},
    {"role": "assistant", "content": "<corrected text>"}
  ],
  "prompt": "Please correct all grammatical errors in ...",
  "source": "<erroneous text>",
  "target": "<corrected text>"
}
```

### Data Augmentation Strategies

1. **Identity Injection** (10%): Reference text as both input and output → teaches "don't over-correct"
2. **Synthetic Noise** (20%): Add controlled noise to reference texts → creates new training pairs
3. **Balanced Sampling** (sqrt): Oversample low-resource languages, downsample high-resource ones

Configuration:
```bash
python -m src.data.augmentation \
    --input processed_data/multigec_train.jsonl \
    --output processed_data/multigec_train_aug.jsonl \
    --identity_ratio 0.1 \
    --noise_ratio 0.2 \
    --noise_rate 0.05 \
    --balance_strategy sqrt  # options: uniform, sqrt, log
```

## Training Configuration

### SFT Stage

- **Model**: Qwen2.5-3B-Instruct
- **Method**: LoRA (r=64, alpha=128)
- **Optimizer**: AdamW, lr=2e-4, cosine schedule
- **Batch size**: 64 (effective, via gradient accumulation)
- **Epochs**: 3
- **Memory optimization**: DeepSpeed ZeRO-2 with CPU optimizer offload

Key config: `configs/sft_config.yaml`

### GRPO Stage

- **Model**: SFT checkpoint (or base model)
- **Group size**: 5 (samples per prompt)
- **Training batch**: 128 prompts
- **Actor lr**: 5e-7
- **KL loss**: enabled (coef=0.01, low_var_kl)
- **Rollout**: vLLM engine, gpu_memory_utilization=0.4
- **Memory optimization**: FSDP with optimizer offload, gradient checkpointing

Key config: `scripts/train_grpo.sh`

### Reward Function

The reward function (`src/rewards/grammar_reward.py`) computes a score in [0, 1]:

1. **Exact match** → 1.0
2. **Edit distance similarity** → Character-level Levenshtein ratio
3. **Length penalty** → Penalizes outputs deviating >30% from reference length
4. **Copy penalty** → 0.3 multiplier if model copies erroneous input unchanged

## Evaluation Metrics

- **Exact Match (EM)**: Fraction of predictions matching reference exactly
- **Edit Distance Similarity (EDS)**: Normalized character-level Levenshtein similarity
- **GLEU**: Sentence-level BLEU variant designed for GEC tasks

All metrics are reported both overall and per-language.

## Directory Structure

```
Multigec/
├── configs/
│   ├── sft_config.yaml      # SFT training configuration
│   └── ds_zero2.json        # DeepSpeed ZeRO-2 config
├── data/                     # Raw MultiGEC data (12 languages)
│   ├── czech/
│   ├── english/
│   └── ...
├── docs/
│   └── OPERATIONS.md         # This document
├── processed_data/           # Generated after data processing
│   ├── multigec_train.jsonl
│   ├── multigec_dev.jsonl
│   ├── multigec_train_aug.jsonl
│   ├── multigec_train_grpo.parquet
│   └── ...
├── scripts/
│   ├── train_sft.py          # SFT training script
│   ├── train_grpo.sh         # GRPO training launch script
│   ├── inference.py          # Batch inference
│   └── run_pipeline.sh       # End-to-end pipeline
├── src/
│   ├── data/
│   │   ├── parse_multigec.py  # Raw data parser
│   │   ├── augmentation.py    # Data augmentation
│   │   └── prepare_verl_data.py # Parquet conversion
│   ├── eval/
│   │   └── evaluate.py        # Evaluation metrics
│   └── rewards/
│       └── grammar_reward.py  # GRPO reward function
├── third_party/
│   └── verl/                  # verl framework (git submodule)
├── requirements.txt
└── README.md
```

## Troubleshooting

### OOM during SFT
- Reduce `per_device_train_batch_size` in `configs/sft_config.yaml`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable CPU optimizer offload in DeepSpeed config (already enabled)

### OOM during GRPO
- Reduce `actor_rollout_ref.rollout.gpu_memory_utilization` (e.g., 0.3)
- Reduce `actor_rollout_ref.rollout.n` (group size, e.g., 3)
- Reduce `data.train_batch_size` (e.g., 64)
- Enable `actor_rollout_ref.actor.fsdp_config.param_offload=True`

### vLLM issues
- Ensure vLLM version is compatible with your CUDA version
- For A30 GPUs, use `--dtype bfloat16`
- If tensor parallel fails, set `tensor_model_parallel_size=1`

### Data parsing issues
- Ensure `pyyaml` is installed for metadata parsing
- Check that all language directories contain proper `.md` files
- Run `python -m src.data.parse_multigec --data_dir data --output_dir processed_data` to see per-corpus stats
