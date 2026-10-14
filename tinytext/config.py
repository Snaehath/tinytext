from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TinyTextConfig:
    """Configuration class for TinyText Transformer architectures.

    Default values correspond to TinyText-Flash-20M-Base (V0 specification).
    """

    model_name: str = "TinyText-Flash-20M-Base"
    vocab_size: int = 8192
    max_seq_len: int = 512
    n_layers: int = 8
    d_model: int = 384
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    intermediate_dim: int = 1408
    activation: str = "swiglu"
    norm: str = "rmsnorm"
    position_encoding: str = "rope"
    attention: str = "grouped_query_attention"
    pre_norm: bool = True
    tie_embeddings: bool = True
    linear_bias: bool = False
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    def __post_init__(self) -> None:
        self.vocab_size = int(self.vocab_size)
        self.max_seq_len = int(self.max_seq_len)
        self.n_layers = int(self.n_layers)
        self.d_model = int(self.d_model)
        self.n_heads = int(self.n_heads)
        self.n_kv_heads = int(self.n_kv_heads)
        self.head_dim = int(self.head_dim)
        self.intermediate_dim = int(self.intermediate_dim)
        self.dropout = float(self.dropout)
        self.rope_theta = float(self.rope_theta)
        self.rms_norm_eps = float(self.rms_norm_eps)
        self.initializer_range = float(self.initializer_range)

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(
                f"head_dim ({self.head_dim}) * n_heads ({self.n_heads}) must equal d_model ({self.d_model})"
            )
        if self.activation.lower() != "swiglu":
            raise ValueError(f"Unsupported activation: {self.activation}. Expected 'swiglu'")
        if self.norm.lower() != "rmsnorm":
            raise ValueError(f"Unsupported norm: {self.norm}. Expected 'rmsnorm'")
        if self.position_encoding.lower() != "rope":
            raise ValueError(
                f"Unsupported position_encoding: {self.position_encoding}. Expected 'rope'"
            )

    def estimate_parameter_count(self) -> int:
        """Calculates exact theoretical parameter count based on configuration."""
        # 1. Token embeddings
        token_embed = self.vocab_size * self.d_model

        # 2. Transformer layers
        # Attention:
        # q_proj: d_model * (n_heads * head_dim)
        # k_proj: d_model * (n_kv_heads * head_dim)
        # v_proj: d_model * (n_kv_heads * head_dim)
        # o_proj: (n_heads * head_dim) * d_model
        # norm: d_model
        q_proj = self.d_model * (self.n_heads * self.head_dim)
        k_proj = self.d_model * (self.n_kv_heads * self.head_dim)
        v_proj = self.d_model * (self.n_kv_heads * self.head_dim)
        o_proj = (self.n_heads * self.head_dim) * self.d_model
        attn_norm = self.d_model

        # SwiGLU MLP:
        # gate_proj: d_model * intermediate_dim
        # up_proj: d_model * intermediate_dim
        # down_proj: intermediate_dim * d_model
        # norm: d_model
        gate_proj = self.d_model * self.intermediate_dim
        up_proj = self.d_model * self.intermediate_dim
        down_proj = self.intermediate_dim * self.d_model
        ffn_norm = self.d_model

        layer_params = (
            q_proj + k_proj + v_proj + o_proj + attn_norm +
            gate_proj + up_proj + down_proj + ffn_norm
        )

        total_layers = self.n_layers * layer_params

        # 3. Final normalization
        final_norm = self.d_model

        # 4. Output LM Head (0 additional params when tied)
        lm_head = 0 if self.tie_embeddings else (self.vocab_size * self.d_model)

        return token_embed + total_layers + final_norm + lm_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TinyTextConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TinyTextConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # If wrapped under a 'model' key, extract it
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
