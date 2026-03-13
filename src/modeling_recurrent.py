from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RecurrentPythiaConfig:
    vocab_size: int = 256
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    ffn_size: int = 2048
    max_position_embeddings: int = 4096
    window_size: Optional[int] = 128
    gru_hidden: int = 256
    use_recurrence: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads


class LayerLocalGRU(nn.Module):
    def __init__(self, hidden_size: int = 512, gru_hidden: int = 256) -> None:
        super().__init__()
        self.gru_hidden = gru_hidden
        self.gate_proj = nn.Linear(hidden_size + gru_hidden, 2 * gru_hidden)
        self.candidate_proj = nn.Linear(hidden_size + gru_hidden, gru_hidden)
        self.out_proj = nn.Linear(gru_hidden, hidden_size)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = x.shape
        if prev_state is None:
            prev_state = torch.zeros(batch, steps, self.gru_hidden, device=x.device, dtype=x.dtype)

        cat_input = torch.cat([x, prev_state], dim=-1)
        z, r = torch.chunk(torch.sigmoid(self.gate_proj(cat_input)), 2, dim=-1)
        candidate = torch.cat([x, r * prev_state], dim=-1)
        new_state = (1.0 - z) * prev_state + z * torch.tanh(self.candidate_proj(candidate))
        return x + self.out_proj(new_state), new_state


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        self.head_dim = head_dim
        self.max_positions = max_positions
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_positions:
            raise ValueError(f"seq_len={seq_len} exceeds rotary max_positions={self.max_positions}")

        positions = torch.arange(seq_len, device=query.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=query.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=query.dtype)
        return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: RecurrentPythiaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.window_size = config.window_size
        self.rotary = RotaryEmbedding(config.head_dim, config.max_position_embeddings)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def _make_attention_mask(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_len, device=device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        mask = key_positions > query_positions
        if self.window_size is not None:
            mask |= (query_positions - key_positions) >= self.window_size
        mask = mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            mask = mask | (~key_padding_mask[:, None, None, :])
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, steps, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, steps, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k, steps)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        mask = self._make_attention_mask(steps, steps, hidden_states.device, attention_mask)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attn_output = torch.matmul(probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, steps, self.hidden_size)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config: RecurrentPythiaConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, config: RecurrentPythiaConfig) -> None:
        super().__init__()
        self.layer_local_gru = LayerLocalGRU(config.hidden_size, config.gru_hidden) if config.use_recurrence else None
        self.pre_attn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.pre_mlp_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        next_state = None
        if self.layer_local_gru is not None:
            hidden_states, next_state = self.layer_local_gru(hidden_states, prev_state)

        hidden_states = hidden_states + self.attn(self.pre_attn_norm(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.mlp(self.pre_mlp_norm(hidden_states))
        return hidden_states, next_state


class RecurrentPythiaForCausalLM(nn.Module):
    def __init__(self, config: RecurrentPythiaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[list[Optional[torch.Tensor]]] = None,
        position_offset: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[Optional[torch.Tensor]]]:
        batch, steps = input_ids.shape
        if steps > self.config.max_position_embeddings:
            raise ValueError(
                f"steps={steps} exceeds max_position_embeddings={self.config.max_position_embeddings}"
            )

        if states is None:
            states = [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError("states length must match num_layers")

        hidden_states = self.embed(input_ids)
        hidden_states = self.dropout(hidden_states)

        next_states: list[Optional[torch.Tensor]] = []
        for layer, prev_state in zip(self.layers, states):
            hidden_states, next_state = layer(hidden_states, prev_state=prev_state, attention_mask=attention_mask)
            next_states.append(next_state)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, next_states
