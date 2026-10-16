from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinytext.config import TinyTextConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inv_freq
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = offset + seq_len
        if self.cos_cached is None or total_len > self.cos_cached.shape[0] or self.cos_cached.device != device:
            self.inv_freq = self.inv_freq.to(device)
            self._build_cache(max(total_len, self.max_seq_len))

        cos = self.cos_cached[offset:offset + seq_len, :].to(device)
        sin = self.sin_cached[offset:offset + seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies RoPE to tensor of shape (batch, n_heads, seq_len, head_dim)."""
    # cos, sin shape: (seq_len, head_dim) -> unsqueeze to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key and value heads across query head groups for Grouped-Query Attention."""
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention (GQA) with RoPE and KV Cache support."""

    def __init__(self, config: TinyTextConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=config.linear_bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.linear_bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=config.linear_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=config.linear_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # KV caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        # Repeat KV heads for GQA
        k_expanded = repeat_kv(k, self.num_kv_groups)
        v_expanded = repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention
        # If cache is used and seq_len == 1, current token can attend to all past keys; no future keys exist.
        is_causal = (past_key_value is None and seq_len > 1)
        dropout_p = self.dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Reshape back to (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        return output, present_key_value


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: TinyTextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_dim, bias=config.linear_bias)
        self.up_proj = nn.Linear(config.d_model, config.intermediate_dim, bias=config.linear_bias)
        self.down_proj = nn.Linear(config.intermediate_dim, config.d_model, bias=config.linear_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer decoder block."""

    def __init__(self, config: TinyTextConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention
        normed_attn = self.attn_norm(hidden_states)
        attn_out, present_kv = self.attn(
            normed_attn,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_out

        # Pre-norm SwiGLU MLP
        normed_ffn = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.mlp(normed_ffn)

        return hidden_states, present_kv


class TinyTextModel(nn.Module):
    """Transformer decoder backbone without LM Head."""

    def __init__(self, config: TinyTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        if position_offset + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length ({position_offset + seq_len}) exceeds max_seq_len ({self.config.max_seq_len})"
            )

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len=seq_len, device=input_ids.device, offset=position_offset)

        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache and present_kv is not None:
                present_key_values.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_key_values


class TinyTextForCausalLM(nn.Module):
    """TinyText Causal Language Model with tied output embeddings."""

    def __init__(self, config: TinyTextConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TinyTextModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        for name, param in self.named_parameters():
            if "weight" in name and "norm" not in name:
                nn.init.normal_(param, mean=0.0, std=std)
            elif "norm" in name and "weight" in name:
                nn.init.ones_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Scale residual projections for stability in deep pre-norm transformers
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for layer in self.model.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=std * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=std * scale)

    def count_parameters(self) -> int:
        """Returns the number of unique trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> Dict[str, Any]:
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_offset=position_offset,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressive text generation supporting KV caching, greedy/sampling, repetition penalty, and EOS stopping."""
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None

        curr_input_ids = input_ids
        for step in range(max_new_tokens):
            if generated.shape[1] >= self.config.max_seq_len:
                break

            position_offset = generated.shape[1] - curr_input_ids.shape[1] if use_cache else 0

            outputs = self.forward(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_offset=position_offset,
            )

            logits = outputs["logits"][:, -1, :].clone()  # (batch, vocab_size)
            if use_cache:
                past_key_values = outputs["past_key_values"]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    unique_tokens = torch.unique(generated[b])
                    logits_b = logits[b, unique_tokens]
                    logits[b, unique_tokens] = torch.where(
                        logits_b > 0,
                        logits_b / repetition_penalty,
                        logits_b * repetition_penalty,
                    )

            # Temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_val = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
                    logits[indices_to_remove] = -float("Inf")

                # Top-p (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if use_cache:
                curr_input_ids = next_token
            else:
                curr_input_ids = generated

            # Check if all batches reached EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
