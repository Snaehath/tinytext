from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytext.config import TinyTextConfig
from tinytext.data import PackedTokenDataset
from tinytext.generate import generate_text, load_model_and_tokenizer
from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer


EVALUATION_PROMPTS = [
    {
        "category": "Prose Continuation",
        "prompt": "The library had stood in the center of the old town for over a century,",
    },
    {
        "category": "Short Story",
        "prompt": "Once upon a time, a small rabbit named Barnaby found a key under an oak tree.",
    },
    {
        "category": "Simple Explanation",
        "prompt": "Photosynthesis is the process that plants use to make food. When sunlight",
    },
    {
        "category": "Descriptive Writing",
        "prompt": "The autumn morning was crisp and quiet, as golden leaves gently drifted",
    },
    {
        "category": "Structured Passage",
        "prompt": "Step 1: Check all hardware connections. Step 2:",
    },
]


@torch.no_grad()
def evaluate_perplexity(
    model: TinyTextForCausalLM,
    dataset_path: str,
    device: torch.device,
    batch_size: int = 4,
    max_batches: int = 50,
) -> Dict[str, float]:
    """Calculates cross-entropy loss and perplexity on a held-out dataset."""
    dataset = PackedTokenDataset(dataset_path, seq_len=model.config.max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(20.0, avg_loss))

    return {
        "cross_entropy_loss": avg_loss,
        "perplexity": perplexity,
        "evaluated_batches": num_batches,
    }


def evaluate_model_comprehensive(
    checkpoint_path: str,
    tokenizer_path: str,
    test_data_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
    device_name: Optional[str] = None,
) -> Dict:
    device = torch.device(device_name) if device_name else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating TinyText model on {device}...")

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, tokenizer_path, device)
    param_count = model.count_parameters()
    weight_size_bytes = Path(checkpoint_path).stat().st_size
    weight_size_mb = weight_size_bytes / (1024 * 1024)

    results = {
        "model_name": model.config.model_name,
        "checkpoint": checkpoint_path,
        "parameters": param_count,
        "weight_size_mb": weight_size_mb,
        "device": str(device),
        "prompts_evaluation": [],
    }

    # 1. Held-out Perplexity
    if test_data_path and Path(test_data_path).exists():
        print(f"Evaluating perplexity on test split {test_data_path}...")
        ppl_metrics = evaluate_perplexity(model, test_data_path, device)
        results["test_metrics"] = ppl_metrics
        print(f"Test Loss: {ppl_metrics['cross_entropy_loss']:.4f} | Perplexity: {ppl_metrics['perplexity']:.2f}")

    # 2. Benchmark Prompt Generations
    print("\n--- Generating on Benchmark Prompts ---")
    for item in EVALUATION_PROMPTS:
        gen = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=item["prompt"],
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            device=device,
        )
        entry = {
            "category": item["category"],
            "prompt": item["prompt"],
            "continuation": gen["continuation"],
            "tokens_per_second": gen["tokens_per_second"],
        }
        results["prompts_evaluation"].append(entry)
        print(f"[{item['category']}]")
        print(f"Prompt: {item['prompt']}")
        print(f"Continuation: {gen['continuation']}\n")

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation report to {out_p}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyText Flash 20M on perplexity and standard prompts")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tinytext_bpe_8k.json")
    parser.add_argument("--test_data", type=str, default="data/processed/test.bin")
    parser.add_argument("--report", type=str, default="reports/evaluation_report.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    evaluate_model_comprehensive(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        test_data_path=args.test_data,
        output_report_path=args.report,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
