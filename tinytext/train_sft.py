from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tinytext.config import TinyTextConfig
from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer
from tinytext.sft_data import SFTDataset, encode_conversation_with_mask
from tinytext.data import PackedTokenDataset


def get_sft_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_sft_loss(model: TinyTextForCausalLM, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Cross entropy loss on assistant targets
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        
        valid_mask = (labels != -100)
        num_valid = valid_mask.sum().item()
        
        if loss is not None and torch.isfinite(loss) and num_valid > 0:
            total_loss += loss.item() * num_valid
            total_tokens += num_valid
            
    model.train()
    return total_loss / max(1, total_tokens)


def train_sft(
    base_checkpoint_path: str = "checkpoints/recovery_400m/best_model.pt",
    train_file: str = "data/instructions/sft_train.jsonl",
    val_file: str = "data/instructions/sft_val.jsonl",
    output_dir: str = "checkpoints/instruct_pilot",
    epochs: int = 1,
    learning_rate: float = 2.0e-5,
    weight_decay: float = 0.01,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    max_seq_len: int = 512,
    warmup_steps: int = 30,
    eval_interval: int = 50,
    log_interval: int = 10,
    max_steps: Optional[int] = None,
    replay_ratio: float = 0.0,
    replay_bin: Optional[str] = None,
    device_name: Optional[str] = None,
):
    root = Path(__file__).resolve().parent.parent
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n==========================================================================================")
    print(f"               TINYTEXT FLASH V1.0 — SUPERVISED INSTRUCTION FINE-TUNING                   ")
    print(f"==========================================================================================")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
    print(f"Base Model    : {base_checkpoint_path}")
    print(f"Output Dir    : {output_dir}")
    print(f"Learning Rate : {learning_rate:.2e} (Warmup: {warmup_steps} steps, Cosine Decay)")
    print(f"Weight Decay  : {weight_decay}")
    print(f"Batch Config  : Micro-batch {micro_batch_size} x Accum {grad_accum_steps} = {micro_batch_size * grad_accum_steps} seqs ({micro_batch_size * grad_accum_steps * max_seq_len:,} tokens/step)")
    print(f"Replay Ratio  : {replay_ratio * 100:.1f}%")
    
    tokenizer = TinyTextTokenizer.load(root / "data/tokenizer/tinytext_bpe_8k.json")
    
    # Load base model
    print(f"\nLoading base model weights from {base_checkpoint_path}...")
    raw = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TinyTextConfig.from_dict(raw["config"])
    model = TinyTextForCausalLM(cfg).to(device)
    model.load_state_dict(raw["model_state_dict"])
    param_count = model.count_parameters()
    print(f"Base model loaded cleanly ({param_count:,} parameters).")
    
    # Load SFT datasets
    print(f"Loading training data from {train_file}...")
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]
        
    train_dataset = SFTDataset(train_data, tokenizer, max_seq_len=max_seq_len)
    val_dataset = SFTDataset(val_data, tokenizer, max_seq_len=max_seq_len)
    
    print(f"Dataset parsed: {len(train_dataset):,} train examples ({train_dataset.total_supervised_tokens:,} supervised tokens), {len(val_dataset):,} val examples.")
    
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False)
    
    # Setup optional pretraining replay
    replay_iter = None
    if replay_ratio > 0.0:
        replay_path = replay_bin or str(root / "data/expanded_shards/train_expanded_shard_00.bin")
        print(f"Enabling pretraining text replay ({replay_ratio * 100:.1f}%) from {replay_path}...")
        replay_ds = PackedTokenDataset(replay_path, seq_len=max_seq_len)
        replay_loader = DataLoader(replay_ds, batch_size=micro_batch_size, shuffle=True)
        replay_iter = iter(replay_loader)
        
    # Optimizer & Scheduler (Fresh for SFT)
    steps_per_epoch = len(train_dataset) // (micro_batch_size * grad_accum_steps)
    total_training_steps = max_steps if max_steps is not None else (steps_per_epoch * epochs)
    
    # Parameter weight decay grouping
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "norm" in n.lower() or "bias" in n.lower():
            nodecay_params.append(p)
        else:
            decay_params.append(p)
            
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    scheduler = get_sft_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
        min_lr_ratio=0.1
    )
    
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Pre-training initial validation evaluation
    print("\nRunning pre-SFT initial validation baseline on held-out instructions...")
    init_val_loss = evaluate_sft_loss(model, val_loader, device)
    init_val_ppl = math.exp(min(20.0, init_val_loss))
    print(f"--> [Pre-SFT Step 0] Validation Loss: {init_val_loss:.4f} | Val Perplexity: {init_val_ppl:.2f}\n")
    
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    best_val_loss = init_val_loss
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    start_time = time.time()
    accum_loss = 0.0
    accum_tokens = 0
    total_tokens_trained = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{epochs} ({steps_per_epoch} optimizer steps) ---")
        
        train_iter = iter(train_loader)
        for _ in range(steps_per_epoch):
            for _ in range(grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                    
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Check replay
                if replay_iter is not None and torch.rand(1).item() < replay_ratio:
                    try:
                        rep_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        rep_batch = next(replay_iter)
                    input_ids = rep_batch["input_ids"].to(device)
                    labels = rep_batch["labels"].to(device)
                    
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(input_ids, labels=labels)
                        loss = outputs["loss"] / grad_accum_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / grad_accum_steps
                    loss.backward()
                    
                accum_loss += loss.item() * grad_accum_steps
                toks_in_b = input_ids.numel()
                accum_tokens += toks_in_b
                total_tokens_trained += toks_in_b
                
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            
            # Logging
            if global_step % log_interval == 0 or global_step == total_training_steps:
                elapsed = time.time() - start_time
                tok_per_sec = accum_tokens / max(1e-5, elapsed)
                current_lr = scheduler.get_last_lr()[0]
                avg_loss = accum_loss / (grad_accum_steps * max(1, log_interval))
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
                
                print(
                    f"Step {global_step:4d}/{total_training_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"Tok/s: {tok_per_sec:7.1f} | "
                    f"VRAM: {vram_mb:5.1f}MB"
                )
                accum_loss = 0.0
                accum_tokens = 0
                start_time = time.time()
                
            # Validation and Checkpointing
            if global_step % eval_interval == 0 or global_step == total_training_steps:
                val_loss = evaluate_sft_loss(model, val_loader, device)
                val_ppl = math.exp(min(20.0, val_loss))
                print(f"--> [Eval Step {global_step}] Validation Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")
                
                # Save latest resumable checkpoint
                latest_ckpt_path = out_dir_path / "latest_checkpoint.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "total_tokens": total_tokens_trained,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": cfg.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, latest_ckpt_path)
                
                # Save best checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt_path = out_dir_path / "best_model.pt"
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "total_tokens": total_tokens_trained,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "config": cfg.to_dict(),
                        "model_state_dict": model.state_dict(),
                    }, best_ckpt_path)
                    print(f"--> Saved new BEST SFT checkpoint to {best_ckpt_path} (val_loss: {best_val_loss:.4f})")
            if global_step >= total_training_steps:
                break
        if global_step >= total_training_steps:
            break
                    
    print(f"\nSFT Completed! Best Validation Loss: {best_val_loss:.4f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train TinyText Flash Instruct model (SFT)")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/recovery_400m/best_model.pt")
    parser.add_argument("--train_file", type=str, default="data/instructions/sft_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/instructions/sft_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/instruct_pilot")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--micro_batch", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--replay_ratio", type=float, default=0.0)
    parser.add_argument("--replay_bin", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_sft(
        base_checkpoint_path=args.base_checkpoint,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        replay_ratio=args.replay_ratio,
        replay_bin=args.replay_bin,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
