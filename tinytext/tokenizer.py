from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


class TinyTextTokenizer:
    """Byte-Level BPE Tokenizer for TinyText.

    Guarantees reversible lossless roundtripping of arbitrary UTF-8 text
    using an 8,192 vocabulary with explicit special tokens.
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

    def __init__(self, tokenizer: Optional[Tokenizer] = None) -> None:
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._build_empty_tokenizer()

    @classmethod
    def _build_empty_tokenizer(cls) -> Tokenizer:
        # ByteLevel BPE base model
        tokenizer = Tokenizer(models.BPE(unk_token=cls.UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> TinyTextTokenizer:
        """Trains a new byte-level BPE tokenizer on an iterator of text documents."""
        raw_tok = cls._build_empty_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        raw_tok.train_from_iterator(iterator, trainer=trainer)
        return cls(raw_tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes string to token IDs."""
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back to a string."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()

    def id_to_token(self, token_id: int) -> Optional[str]:
        return self._tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tokenizer.token_to_id(token)

    def save(self, path: Union[str, Path]) -> None:
        """Saves tokenizer configuration and vocabulary to a json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> Union[TinyTextTokenizer, Any]:
        """Loads tokenizer from saved json file (BPE or Byte-level)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        # Check if byte-level specification
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = json.load(f)
            if isinstance(header, dict) and header.get("type") == "byte_level_260":
                from tinytext.tokenizer_byte import TinyTextByteTokenizer
                return TinyTextByteTokenizer()
        except Exception:
            pass

        raw_tok = Tokenizer.from_file(str(path))
        return cls(raw_tok)
