"""Isolated RLVR Pilot Implementation for TinyText Flash.
Implements Group-Relative Policy Optimization (GRPO / Group-Relative REINFORCE):
- Generates K = 4 completions per prompt.
- Computes verifiable deterministic reward R_total.
- Computes group-normalized advantages A_i = (R_i - mean(R)) / (std(R) + 1e-4).
- Applies policy gradient loss on generated target tokens.
- Tracks loss, reward mean/std, peak VRAM, and gradient norm.
"""

import math
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from tinytext.config import TinyTextConfig
from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer
from tinytext.generate import load_model_and_tokenizer
from scripts.test_rlvr_reward import compute_verifiable_reward


def run_rlvr_pilot(
    base_checkpoint_path: str = "checkpoints/instruct_replay/best_model.pt",
    output_dir: str = "checkpoints/rlvr_pilot",
    num_steps: int = 15,
    group_size: int = 4,
    learning_rate: float = 1.0e-6,
    max_gen_tokens: int = 20,
    device_name: str = "cuda",
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"       TINYTEXT FLASH V1.2 — ISOLATED RLVR PILOT       ")
    print(f"=======================================================")
    print(f"Device:           {device}")
    print(f"Base Checkpoint:  {base_checkpoint_path}")
    print(f"Group Size (K):   {group_size}")
    print(f"Learning Rate:    {learning_rate}")
    print(f"Steps:            {num_steps}")

    tok_path = "data/tokenizer/tinytext_bpe_8k.json"
    tokenizer = TinyTextTokenizer.load(tok_path)

    raw = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    config = TinyTextConfig.from_dict(raw["config"])
    model = TinyTextForCausalLM(config).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    train_prompts = [
        ("What is 2 + 3? Answer with only the number.", "5"),
        ("Compute 4 + 4. Output only the number.", "8"),
        ("What is 10 - 5? Answer with only the number.", "5"),
        ("Calculate 6 + 7. Answer with only the number.", "13"),
        ("What is 3 * 3? Output only the number.", "9"),
        ("Compute 8 - 3. Output only the number.", "5"),
        ("What is 12 / 4? Answer with only the number.", "3"),
        ("Calculate 15 - 8. Answer with only the number.", "7"),
    ]

    template = (
        "System:\n"
        "You are TinyText, a helpful and concise assistant.\n\n"
        "User:\n"
        "{user_message}\n\n"
        "Assistant:\n"
    )

    history = []

    for step in range(1, num_steps + 1):
        prompt_idx = (step - 1) % len(train_prompts)
        user_q, target = train_prompts[prompt_idx]
        full_p = template.format(user_message=user_q)

        p_ids = tokenizer.encode(full_p, add_special_tokens=False)
        p_len = len(p_ids)
        p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)

        # 1. Rollout group of K candidates with sampling
        model.eval()
        group_ids = []
        group_rewards = []
        completions = []

        with torch.no_grad():
            for k in range(group_size):
                gen_ids = model.generate(
                    p_tensor,
                    max_new_tokens=max_gen_tokens,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    eos_token_id=tokenizer.EOS_ID,
                    use_cache=True,
                )[0]
                group_ids.append(gen_ids)

                full_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
                continuation = full_text[len(full_p):].replace("<|endoftext|>", "").strip()
                completions.append(continuation)

                r, _ = compute_verifiable_reward(continuation, target)
                group_rewards.append(r)

        # 2. Compute group-relative advantages
        r_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std(unbiased=False)
        advantages = (r_tensor - r_mean) / (r_std + 1.0e-4)

        # 3. Policy gradient update on generated tokens
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        # Compute log-probs on generated tokens
        for k in range(group_size):
            cand_tensor = group_ids[k].unsqueeze(0)  # (1, seq_len)
            if cand_tensor.shape[1] <= p_len:
                continue

            out = model(cand_tensor)
            logits = out["logits"][:, :-1, :]  # (1, seq_len-1, vocab)
            targets = cand_tensor[:, 1:]        # (1, seq_len-1)

            # Log-probs of targets
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

            # Mask only generated positions (after prompt)
            gen_log_probs = token_log_probs[:, p_len - 1:]

            # Policy gradient loss: -A * sum(log_prob)
            adv = advantages[k]
            cand_loss = -adv * gen_log_probs.sum()
            total_loss += cand_loss / group_size

        if torch.isfinite(total_loss) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
        step_record = {
            "step": step,
            "prompt": user_q,
            "target": target,
            "rewards": [round(float(x), 2) for x in group_rewards],
            "mean_reward": round(float(r_mean.item()), 4),
            "std_reward": round(float(r_std.item()), 4),
            "loss": round(float(total_loss.item()), 4),
            "peak_vram_mb": round(peak_vram, 2),
            "samples": completions,
        }
        history.append(step_record)

        print(f"Step {step:2d}/{num_steps} | Loss: {total_loss.item():.4f} | R_mean: {r_mean.item():.2f} (std: {r_std.item():.2f}) | VRAM: {peak_vram:.1f}MB")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "rlvr_history": history,
    }, out_p / "rlvr_checkpoint.pt")

    print(f"\nRLVR Pilot Completed. Checkpoint saved to: {out_p / 'rlvr_checkpoint.pt'}")
    return history


def main():
    run_rlvr_pilot(num_steps=15)


if __name__ == "__main__":
    main()
