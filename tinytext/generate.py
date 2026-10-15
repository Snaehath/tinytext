from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytext.config import TinyTextConfig
from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer


def load_model_and_tokenizer(
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device,
) -> tuple[TinyTextForCausalLM, TinyTextTokenizer]:
    """Loads model from checkpoint and tokenizer from file."""
    tok_path = Path(tokenizer_path)
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    tokenizer = TinyTextTokenizer.load(tok_path)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = TinyTextConfig.from_dict(checkpoint.get("config", {}))

    model = TinyTextForCausalLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def generate_text(
    model: TinyTextForCausalLM,
    tokenizer: TinyTextTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    use_cache: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) == 0:
        input_ids = [tokenizer.BOS_ID]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    generated_tensor = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.EOS_ID,
        use_cache=use_cache,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    total_new_tokens = generated_tensor.shape[1] - input_tensor.shape[1]
    tok_per_sec = total_new_tokens / max(1e-5, elapsed)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    generated_ids = generated_tensor[0].tolist()
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
    continuation = full_text[len(prompt_decoded):] if full_text.startswith(prompt_decoded) else full_text

    return {
        "prompt": prompt,
        "continuation": continuation,
        "full_text": full_text,
        "tokens_generated": total_new_tokens,
        "time_seconds": elapsed,
        "tokens_per_second": tok_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "use_cache": use_cache,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text using TinyText Flash 20M Base model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/tinytext_bpe_8k.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cache", action="store_true", help="Disable KV caching to compare against uncached mode")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--orchestrate", action="store_true", help="Run via ReasoningOrchestrator")
    parser.add_argument(
        "--refinement",
        type=str,
        default="none",
        choices=["none", "generic_critique", "constraint_guided"],
        help="Test-time refinement strategy (default: none)",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device)

    print(f"\n--- Prompt ---\n{args.prompt}")

    if args.orchestrate or args.refinement != "none":
        from tinytext.orchestrator import ReasoningOrchestrator
        orch = ReasoningOrchestrator(model=model, tokenizer=tokenizer, device=device)
        res = orch.process(
            user_query=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            refinement_strategy=args.refinement,
        )
        print(f"\n--- Output (Route: {res.route.value}, Status: {res.status.value}) ---\n{res.output}")
        print("\n--- Performance Metrics ---")
        print(f"Verified         : {res.verified}")
        print(f"Task Success     : {res.task_success}")
        if res.tool_used:
            print(f"Tool Used        : {res.tool_used}")
        print(f"Tokens generated : {res.tokens_generated}")
        print(f"Latency          : {res.latency_seconds:.3f} s")
    else:
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
            device=device,
        )

        print(f"\n--- Generated Continuation ---\n{result['continuation']}")
        print("\n--- Performance Metrics ---")
        print(f"Tokens generated : {result['tokens_generated']}")
        print(f"Latency          : {result['time_seconds']:.3f} s")
        print(f"Throughput       : {result['tokens_per_second']:.2f} tokens/sec")
        print(f"Mode             : {'Cached (KV-Cache)' if result['use_cache'] else 'Uncached'}")
        if device.type == "cuda":
            print(f"Peak VRAM        : {result['peak_vram_mb']:.2f} MB")


if __name__ == "__main__":
    main()
