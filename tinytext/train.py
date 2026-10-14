from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytext.config import TinyTextConfig
from tinytext.data import PackedTokenDataset
from tinytext.model import TinyTextForCausalLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine learning rate decay with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_recovery_schedule(
    optimizer: torch.optim.Optimizer,
    start_step: int,
    total_steps: int,
    initial_lr: float = 7.5e-5,
    peak_lr: float = 1.2e-4,
    min_lr: float = 2.0e-5,
    micro_warmup_steps: int = 300,
) -> LambdaLR:
    """Smooth recovery schedule preventing learning rate jumps upon resuming.
    
    Starts exactly at initial_lr (e.g. 7.5e-5), micro-warms up over 300 steps
    to peak_lr (1.2e-4), then smoothly decays to min_lr (2.0e-5) at total_steps.
    """
    base_opt_lr = optimizer.defaults["lr"]
    init_ratio = initial_lr / base_opt_lr
    peak_ratio = peak_lr / base_opt_lr
    min_ratio = min_lr / base_opt_lr

    def lr_lambda(step: int) -> float:
        if step < start_step:
            return init_ratio
        offset = step - start_step
        if offset < micro_warmup_steps:
            progress = offset / max(1, micro_warmup_steps)
            return init_ratio + (peak_ratio - init_ratio) * progress
        decay_steps = total_steps - start_step - micro_warmup_steps
        decay_offset = offset - micro_warmup_steps
        progress = min(1.0, max(0.0, decay_offset / max(1, decay_steps)))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (peak_ratio - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    checkpoint_path: Path,
    model: TinyTextForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    step: int,
    total_tokens: int,
    best_val_loss: float,
    config: TinyTextConfig,
    train_args: Dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "total_tokens": total_tokens,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
        "train_args": train_args,
        "rng_states": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: TinyTextForCausalLM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int, float]:
    """Loads checkpoint and returns (step, total_tokens, best_val_loss)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        if "torch" in rng and rng["torch"] is not None:
            try:
                torch_state = rng["torch"]
                if not isinstance(torch_state, torch.ByteTensor):
                    torch_state = torch_state.cpu().to(torch.uint8)
                torch.set_rng_state(torch_state)
            except Exception:
                pass
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            try:
                cuda_states = [s.cpu().to(torch.uint8) if not isinstance(s, torch.ByteTensor) else s for s in rng["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
        if "numpy" in rng and rng["numpy"] is not None:
            try:
                np.random.set_state(rng["numpy"])
            except Exception:
                pass
        if "python" in rng and rng["python"] is not None:
            try:
                random.setstate(rng["python"])
            except Exception:
                pass

    step = checkpoint.get("step", 0)
    total_tokens = checkpoint.get("total_tokens", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return step, total_tokens, best_val_loss


@torch.no_grad()
def evaluate_loss(
    model: TinyTextForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_steps: int = 250,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0

    for batch in val_loader:
        if steps >= max_eval_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids, labels=labels)
        else:
            outputs = model(input_ids, labels=labels)

        loss = outputs["loss"]
        if loss is not None and torch.isfinite(loss):
            total_loss += loss.item()
            steps += 1

    model.train()
    return total_loss / max(1, steps)


def train(
    config_path: str = "configs/flash_20m.yaml",
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    max_steps: Optional[int] = None,
    eval_interval: Optional[int] = None,
    log_interval: Optional[int] = None,
    device_name: Optional[str] = None,
    recovery_schedule: bool = False,
) -> None:
    config = TinyTextConfig.from_yaml(config_path)

    # Device selection
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    set_seed(42)

    # Initialize model
    model = TinyTextForCausalLM(config).to(device)
    param_count = model.count_parameters()
    print(f"Initialized {config.model_name} with {param_count:,} trainable parameters.")

    # Training settings
    micro_batch_size = 4
    effective_tokens_target = 16384
    tokens_per_micro_batch = micro_batch_size * config.max_seq_len  # 4 * 512 = 2048
    grad_accum_steps = max(1, effective_tokens_target // tokens_per_micro_batch)  # 8
    lr = 3.0e-4
    weight_decay = 0.1
    grad_clip = 1.0
    total_steps = max_steps or 2000
    warmup = min(500, total_steps // 10)
    eval_every = eval_interval or 200
    log_every = log_interval or 20

    print(f"Effective Batch: {grad_accum_steps * tokens_per_micro_batch} tokens ({grad_accum_steps} accum steps x {micro_batch_size} batch x {config.max_seq_len} seq_len)")

    # Optimizer with weight decay exclusion for 1D biases and RMSNorm
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = get_cosine_schedule(optimizer, warmup_steps=warmup, max_steps=total_steps, min_lr_ratio=0.1)

    # AMP setup
    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    # Resume checkpoint if provided
    start_step = 0
    total_tokens_trained = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        start_step, total_tokens_trained, best_val_loss = load_checkpoint(
            Path(resume_checkpoint), model, optimizer, scheduler if not recovery_schedule else None, scaler, device
        )
        if recovery_schedule:
            scheduler = get_recovery_schedule(
                optimizer,
                start_step=start_step,
                total_steps=total_steps,
                initial_lr=7.5e-5,
                peak_lr=1.2e-4,
                min_lr=2.0e-5,
                micro_warmup_steps=300,
            )
            scheduler.last_epoch = start_step
        print(f"Pre-flight Resume State: Global Step={start_step:,} | Processed Tokens={total_tokens_trained:,} | Current LR={scheduler.get_last_lr()[0]:.2e}")

    # Dataset loading
    train_path = train_data_path or "data/processed/train.bin"
    val_path = val_data_path or "data/processed/val.bin"

    if not Path(train_path).exists():
        print(f"Train data {train_path} not found. Preparing fallback synthetic dataset for training pipeline...")
        from tinytext.data import pack_and_save_tokens
        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        dummy_seqs = [list(range(10, 500)) for _ in range(64)]
        pack_and_save_tokens(dummy_seqs, train_path, seq_len=config.max_seq_len)
        pack_and_save_tokens(dummy_seqs[:16], val_path, seq_len=config.max_seq_len)

    from tinytext.data import ShardedPackedTokenDataset
    train_dataset = ShardedPackedTokenDataset(train_path, seq_len=config.max_seq_len)
    val_dataset = PackedTokenDataset(val_path, seq_len=config.max_seq_len)
    num_shards = len(train_dataset.shards) if hasattr(train_dataset, "shards") else 1
    print(f"Loaded training data: {len(train_dataset):,} sequences ({train_dataset.total_tokens:,} tokens across {num_shards} shard(s)).")

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False)

    train_iter = iter(train_loader)

    print("Beginning training loop...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    step = start_step
    accum_loss = 0.0
    start_time = time.time()
    accum_tokens = 0

    while step < total_steps:
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()

            accum_loss += loss.item() * grad_accum_steps
            tokens_in_batch = input_ids.numel()
            accum_tokens += tokens_in_batch
            total_tokens_trained += tokens_in_batch

        # Gradient clipping and optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        # Logging
        if step % log_every == 0 or step == total_steps:
            elapsed = time.time() - start_time
            tok_per_sec = accum_tokens / max(1e-5, elapsed)
            step_per_sec = log_every / max(1e-5, elapsed)
            current_lr = scheduler.get_last_lr()[0]
            steps_logged = log_every if step % log_every == 0 else (step % log_every)
            avg_loss = accum_loss / (grad_accum_steps * max(1, steps_logged))
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

            print(
                f"Step {step:5d}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"GradNorm: {grad_norm:.2f} | "
                f"Tok/s: {tok_per_sec:7.1f} | "
                f"Tokens: {total_tokens_trained:,} | "
                f"VRAM: {vram_mb:5.1f}MB"
            )

            accum_loss = 0.0
            accum_tokens = 0
            start_time = time.time()

        # Validation and Checkpointing
        if step % eval_every == 0 or step == total_steps:
            val_loss = evaluate_loss(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
            val_ppl = math.exp(min(20.0, val_loss))
            print(f"--> [Eval Step {step}] Validation Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

            # Save latest resumable checkpoint
            save_checkpoint(
                ckpt_dir / "latest_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                total_tokens=total_tokens_trained,
                best_val_loss=best_val_loss,
                config=config,
                train_args={"lr": lr, "effective_tokens_target": effective_tokens_target},
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    total_tokens=total_tokens_trained,
                    best_val_loss=best_val_loss,
                    config=config,
                    train_args={"lr": lr, "effective_tokens_target": effective_tokens_target},
                )
                print(f"--> Saved new BEST checkpoint to {ckpt_dir / 'best_model.pt'} (val_loss: {best_val_loss:.4f})")

    print(f"Training completed successfully! Total tokens processed: {total_tokens_trained:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyText Flash 20M model")
    parser.add_argument("--config", type=str, default="configs/flash_20m.yaml")
    parser.add_argument("--train_data", "--train_data_path", type=str, default=None, help="Path to train .bin file or shard directory")
    parser.add_argument("--val_data", "--val_data_path", type=str, default=None, help="Path to val .bin file")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--recovery_schedule", action="store_true", help="Use smooth recovery schedule without learning rate jumps")
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        resume_checkpoint=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        device_name=args.device,
        recovery_schedule=args.recovery_schedule,
    )


if __name__ == "__main__":
    main()
