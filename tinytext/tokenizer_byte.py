from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class TinyTextByteTokenizer:
    """Direct Byte-Level Tokenizer for TinyText.

    Maps 256 raw byte values (0x00 - 0xFF) to token IDs 4..259,
    with 4 reserved special tokens at IDs 0..3:
      0: <|endoftext|> (EOS)
      1: <|pad|>        (PAD)
      2: <|bos|>        (BOS)
      3: <|unk|>        (UNK)

    Total vocabulary size: exactly 260 token IDs.
    Guarantees 100% reversible, lossless roundtripping of any arbitrary byte sequence.
    """

    SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|unk|>"]
    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    UNK_TOKEN = "<|unk|>"

    EOS_ID = 0
    PAD_ID = 1
    BOS_ID = 2
    UNK_ID = 3

    NUM_SPECIAL = 4
    BYTE_OFFSET = 4
    TOTAL_VOCAB_SIZE = 260

    def __init__(self) -> None:
        # Build token <-> id mappings
        self._id_to_token: Dict[int, str] = {
            self.EOS_ID: self.EOS_TOKEN,
            self.PAD_ID: self.PAD_TOKEN,
            self.BOS_ID: self.BOS_TOKEN,
            self.UNK_ID: self.UNK_TOKEN,
        }
        for b in range(256):
            tid = self.BYTE_OFFSET + b
            # Canonical byte representation <0xXX>
            self._id_to_token[tid] = f"<0x{b:02X}>"

        self._token_to_id: Dict[str, int] = {v: k for k, v in self._id_to_token.items()}

    @property
    def vocab_size(self) -> int:
        return self.TOTAL_VOCAB_SIZE

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._id_to_token.get(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._token_to_id.get(token)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string to raw UTF-8 byte token IDs."""
        raw_bytes = text.encode("utf-8")
        ids = [b + self.BYTE_OFFSET for b in raw_bytes]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def encode_bytes(self, data: bytes, add_special_tokens: bool = False) -> List[int]:
        """Encodes raw bytes directly to token IDs."""
        ids = [b + self.BYTE_OFFSET for b in data]
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a UTF-8 string."""
        chunks: List[str] = []
        byte_buffer = bytearray()

        for tid in ids:
            if 0 <= tid < self.BYTE_OFFSET:
                if byte_buffer:
                    chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                    byte_buffer = bytearray()
                if not skip_special_tokens:
                    chunks.append(self.SPECIAL_TOKENS[tid])
            elif self.BYTE_OFFSET <= tid < self.TOTAL_VOCAB_SIZE:
                byte_buffer.append(tid - self.BYTE_OFFSET)
            else:
                # Out-of-range token ID
                if not skip_special_tokens:
                    chunks.append(self.UNK_TOKEN)

        if byte_buffer:
            chunks.append(byte_buffer.decode("utf-8", errors="replace"))

        return "".join(chunks)

    def decode_to_bytes(self, ids: List[int], skip_special_tokens: bool = True) -> bytes:
        """Decodes token IDs directly to raw bytes."""
        byte_buffer = bytearray()
        for tid in ids:
            if self.BYTE_OFFSET <= tid < self.TOTAL_VOCAB_SIZE:
                byte_buffer.append(tid - self.BYTE_OFFSET)
            elif not skip_special_tokens and 0 <= tid < self.BYTE_OFFSET:
                pass
        return bytes(byte_buffer)

    def save(self, path: Union[str, Path]) -> None:
        """Saves tokenizer metadata and vocabulary specification to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "byte_level_260",
            "vocab_size": self.TOTAL_VOCAB_SIZE,
            "byte_offset": self.BYTE_OFFSET,
            "special_tokens": {
                self.EOS_TOKEN: self.EOS_ID,
                self.PAD_TOKEN: self.PAD_ID,
                self.BOS_TOKEN: self.BOS_ID,
                self.UNK_TOKEN: self.UNK_ID,
            },
            "vocab": self._token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> TinyTextByteTokenizer:
        """Loads byte tokenizer from a saved specification file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Byte tokenizer file not found: {path}")
        # The mapping is deterministic, but verify file integrity
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "byte_level_260" or data.get("vocab_size") != cls.TOTAL_VOCAB_SIZE:
            raise ValueError(f"Invalid byte tokenizer specification in {path}")
        return cls()
