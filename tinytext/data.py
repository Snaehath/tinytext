from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_text(text: str) -> str:
    """Conservatively normalizes whitespace without altering valid Unicode or punctuation."""
    # Replace carriage returns and replace tabs with spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading and trailing whitespace
    return text.strip()


def is_repetitive(text: str, max_repeat_ratio: float = 0.3) -> bool:
    """Checks if text contains excessively repetitive n-grams or lines."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 4:
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < (1.0 - max_repeat_ratio):
            return True

    words = text.split()
    if len(words) > 30:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.2:
            return True

    return False


def clean_and_filter_document(
    text: str,
    min_chars: int = 50,
    max_chars: int = 50000,
) -> Optional[str]:
    """Filters empty, malformed, or excessively repetitive text."""
    if not isinstance(text, str):
        return None

    cleaned = normalize_text(text)
    if len(cleaned) < min_chars or len(cleaned) > max_chars:
        return None

    if is_repetitive(cleaned):
        return None

    return cleaned


def compute_doc_hash(text: str) -> str:
    """Computes SHA-256 hash for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PackedTokenDataset(Dataset):
    """Memory-mapped dataset of fixed-length token sequences for pretraining."""

    def __init__(self, bin_path: Union[str, Path], seq_len: int = 512) -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len

        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token file not found: {self.bin_path}")

        file_bytes = self.bin_path.stat().st_size
        item_size = 2  # uint16 is 2 bytes
        self.total_tokens = file_bytes // item_size

        if self.total_tokens < self.seq_len:
            raise ValueError(
                f"File {self.bin_path} contains {self.total_tokens} tokens, which is fewer than seq_len ({self.seq_len})"
            )

        self.num_sequences = self.total_tokens // self.seq_len
        # Memory-map the binary file for zero-copy read access
        self.data = np.memmap(
            self.bin_path,
            dtype=np.uint16,
            mode="r",
            shape=(self.num_sequences, self.seq_len),
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = np.array(self.data[idx], dtype=np.int64)
        tensor = torch.from_numpy(tokens)
        return {
            "input_ids": tensor,
            "labels": tensor,
        }


class ShardedPackedTokenDataset(Dataset):
    """Memory-mapped dataset aggregating multiple binary token shards seamlessly."""

    def __init__(self, path_or_dir: Union[str, Path, List[Path]], seq_len: int = 512) -> None:
        self.seq_len = seq_len
        self.shards: List[PackedTokenDataset] = []
        self.cumulative_sizes: List[int] = []

        if isinstance(path_or_dir, list):
            shard_paths = [Path(p) for p in path_or_dir]
        else:
            p = Path(path_or_dir)
            if p.is_dir():
                shard_paths = sorted(list(p.glob("*.bin")))
            else:
                shard_paths = [p]

        if not shard_paths:
            raise FileNotFoundError(f"No .bin shards found at {path_or_dir}")

        total_seqs = 0
        for sp in shard_paths:
            ds = PackedTokenDataset(sp, seq_len=seq_len)
            self.shards.append(ds)
            total_seqs += len(ds)
            self.cumulative_sizes.append(total_seqs)

        self.total_sequences = total_seqs
        self.total_tokens = sum(s.total_tokens for s in self.shards)

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import bisect
        shard_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if shard_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[shard_idx - 1]
        return self.shards[shard_idx][local_idx]


def pack_and_save_tokens(
    token_lists: Iterable[List[int]],
    output_bin_path: Union[str, Path],
    seq_len: int = 512,
    eos_token_id: int = 0,
) -> int:
    """Packs sequences of tokens separated by EOS into fixed-length blocks and writes to binary uint16.

    Returns total number of sequences written.
    """
    output_bin_path = Path(output_bin_path)
    output_bin_path.parent.mkdir(parents=True, exist_ok=True)

    buffer: List[int] = []
    num_written = 0

    with open(output_bin_path, "wb") as f:
        for tokens in token_lists:
            buffer.extend(tokens)
            buffer.append(eos_token_id)

            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                arr = np.array(chunk, dtype=np.uint16)
                f.write(arr.tobytes())
                num_written += 1

    return num_written
