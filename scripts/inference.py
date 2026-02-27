"""
Batch inference script for grammar correction models.

Supports:
  - HuggingFace models (with optional LoRA adapter)
  - vLLM for fast batched generation

Usage:
    # Standard HuggingFace inference
    python scripts/inference.py \
        --model_path outputs/sft/final \
        --input processed_data/multigec_dev.jsonl \
        --output outputs/predictions_dev.jsonl

    # With vLLM acceleration
    python scripts/inference.py \
        --model_path outputs/sft/final \
        --input processed_data/multigec_dev.jsonl \
        --output outputs/predictions_dev.jsonl \
        --use_vllm

    # Using base model + LoRA adapter
    python scripts/inference.py \
        --model_path Qwen/Qwen2.5-3B-Instruct \
        --lora_path outputs/sft/final \
        --input processed_data/multigec_dev.jsonl \
        --output outputs/predictions_dev.jsonl
"""

import json
import argparse
from tqdm import tqdm


def load_samples(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def inference_hf(model_path, lora_path, samples, max_new_tokens, temperature, batch_size):
    """HuggingFace Transformers inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()

    results = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[i:i + batch_size]
        prompts = []
        for s in batch:
            messages = [{"role": "user", "content": s["prompt"]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, s in enumerate(batch):
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[j][input_len:]
            pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append({
                "id": s["id"],
                "lang": s["lang"],
                "source": s["source"],
                "prediction": pred,
            })

    return results


def inference_vllm(model_path, lora_path, samples, max_new_tokens, temperature):
    """vLLM batched inference."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs)

    prompts = []
    for s in samples:
        messages = [{"role": "user", "content": s["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for s, output in zip(samples, outputs):
        pred = output.outputs[0].text.strip()
        results.append({
            "id": s["id"],
            "lang": s["lang"],
            "source": s["source"],
            "prediction": pred,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_vllm", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    if args.use_vllm:
        results = inference_vllm(args.model_path, args.lora_path, samples, args.max_new_tokens, args.temperature)
    else:
        results = inference_hf(args.model_path, args.lora_path, samples, args.max_new_tokens, args.temperature, args.batch_size)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
