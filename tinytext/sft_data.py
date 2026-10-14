from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from tinytext.tokenizer import TinyTextTokenizer

SYSTEM_PROMPT_DEFAULT = (
    "You are TinyText, a helpful and concise assistant. Follow the user's instructions, "
    "answer accurately when possible, and acknowledge uncertainty when needed."
)


def format_conversation(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> Tuple[str, List[Tuple[int, int]]]:
    """Formats a multi-turn conversation into standard text and returns character spans for assistant responses.

    Format:
    System:
    {system_prompt}

    User:
    {user_message_1}

    Assistant:
    {assistant_response_1}<|endoftext|>
    """
    sys_text = system_prompt or SYSTEM_PROMPT_DEFAULT
    full_text = f"System:\n{sys_text.strip()}\n\n"
    
    assistant_spans: List[Tuple[int, int]] = []
    
    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()
        
        if role == "user":
            full_text += f"User:\n{content}\n\n"
        elif role == "assistant":
            prefix = "Assistant:\n"
            full_text += prefix
            start_idx = len(full_text)
            full_text += content + "<|endoftext|>\n\n"
            end_idx = len(full_text)
            assistant_spans.append((start_idx, end_idx))
            
    return full_text.rstrip("\n"), assistant_spans


def encode_conversation_with_mask(
    messages: List[Dict[str, str]],
    tokenizer: TinyTextTokenizer,
    max_seq_len: int = 512,
    system_prompt: Optional[str] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Encodes a conversation and generates input_ids and labels with assistant-only loss masking.

    Non-assistant tokens (System prompt, User messages, role headers, padding) receive -100 in labels.
    """
    sys_text = system_prompt or SYSTEM_PROMPT_DEFAULT
    
    token_ids: List[int] = []
    labels: List[int] = []
    
    # 1. System Prompt (Masked)
    sys_header = f"System:\n{sys_text.strip()}\n\n"
    sys_toks = tokenizer.encode(sys_header, add_special_tokens=False)
    token_ids.extend(sys_toks)
    labels.extend([-100] * len(sys_toks))
    
    for i, msg in enumerate(messages):
        role = msg["role"].strip().lower()
        content = msg["content"].strip()
        
        if role == "user":
            user_text = f"User:\n{content}\n\n"
            user_toks = tokenizer.encode(user_text, add_special_tokens=False)
            token_ids.extend(user_toks)
            labels.extend([-100] * len(user_toks))
            
        elif role == "assistant":
            prefix_text = "Assistant:\n"
            prefix_toks = tokenizer.encode(prefix_text, add_special_tokens=False)
            token_ids.extend(prefix_toks)
            labels.extend([-100] * len(prefix_toks))
            
            # Response tokens + EOS
            resp_toks = tokenizer.encode(content, add_special_tokens=False)
            resp_toks.append(tokenizer.EOS_ID)
            
            token_ids.extend(resp_toks)
            labels.extend(resp_toks)  # Supervised target!
            
            # Add trailing newline for formatting if not last
            if i < len(messages) - 1:
                sep_toks = tokenizer.encode("\n\n", add_special_tokens=False)
                token_ids.extend(sep_toks)
                labels.extend([-100] * len(sep_toks))
                
    # Truncate if too long (conservatively preserving up to max_seq_len)
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        
    # Check that at least some assistant targets survived truncation
    num_supervised = sum(1 for x in labels if x != -100)
    if num_supervised == 0:
        return None
        
    pad_len = max_seq_len - len(token_ids)
    attention_mask = [1] * len(token_ids) + [0] * pad_len
    
    # Pad to max_seq_len
    token_ids.extend([tokenizer.PAD_ID] * pad_len)
    labels.extend([-100] * pad_len)
    
    return {
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "num_supervised_tokens": num_supervised,
    }


class SFTDataset(Dataset):
    """PyTorch Dataset for Supervised Instruction Fine-Tuning."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: TinyTextTokenizer, max_seq_len: int = 512, system_prompt: Optional[str] = None):
        self.examples = []
        self.total_supervised_tokens = 0
        
        for item in data:
            messages = item["messages"]
            encoded = encode_conversation_with_mask(messages, tokenizer, max_seq_len=max_seq_len, system_prompt=system_prompt)
            if encoded is not None:
                self.examples.append(encoded)
                self.total_supervised_tokens += encoded["num_supervised_tokens"]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]
