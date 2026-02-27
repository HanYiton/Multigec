"""
SFT training script for MultiGEC grammar correction.

Usage:
    torchrun --nproc_per_node=2 scripts/train_sft.py --config configs/sft_config.yaml
    # or with deepspeed:
    deepspeed --num_gpus=2 scripts/train_sft.py --config configs/sft_config.yaml
"""

import json
import yaml
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def build_dataset(samples, tokenizer, max_length: int) -> Dataset:
    """Convert chat samples to tokenized dataset for SFT."""
    input_ids_list = []
    labels_list = []

    for s in samples:
        messages = s["messages"]
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        ids = encoded["input_ids"].squeeze(0)

        # Build labels: mask the user turn (only train on assistant response)
        user_text = tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        user_len = len(tokenizer(user_text, truncation=True, max_length=max_length)["input_ids"])

        labels = ids.clone()
        labels[:user_len] = -100  # mask user turn

        input_ids_list.append(ids)
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        attn_implementation=cfg["model"].get("attn_implementation"),
        trust_remote_code=True,
    )

    # Apply LoRA
    if cfg["lora"]["enabled"]:
        lora_config = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["lora_alpha"],
            lora_dropout=cfg["lora"]["lora_dropout"],
            target_modules=cfg["lora"]["target_modules"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    train_samples = load_jsonl(cfg["data"]["train_file"])
    val_samples = load_jsonl(cfg["data"]["val_file"])
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    max_length = cfg["data"]["max_length"]
    train_dataset = build_dataset(train_samples, tokenizer, max_length)
    val_dataset = build_dataset(val_samples, tokenizer, max_length)

    # Training arguments
    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy=tcfg["eval_strategy"],
        eval_steps=tcfg["eval_steps"],
        save_strategy=tcfg["save_strategy"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        load_best_model_at_end=tcfg["load_best_model_at_end"],
        metric_for_best_model=tcfg["metric_for_best_model"],
        greater_is_better=tcfg["greater_is_better"],
        dataloader_num_workers=tcfg["dataloader_num_workers"],
        report_to=tcfg.get("report_to", "none"),
        run_name=tcfg.get("run_name"),
        deepspeed=tcfg.get("deepspeed"),
        seed=tcfg.get("seed", 42),
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    trainer.train()
    trainer.save_model(tcfg["output_dir"] + "/final")
    tokenizer.save_pretrained(tcfg["output_dir"] + "/final")
    print(f"Model saved to {tcfg['output_dir']}/final")


if __name__ == "__main__":
    main()
