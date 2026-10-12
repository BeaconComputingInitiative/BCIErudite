"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Decoder-Only Transformer Engine (RoPE + RMSNorm + SwiGLU + KV-Cache + SDPA + Top-P)
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, seq_len: int, start_pos: int = 0):
        # Slicing succeeds if and only if sequence bounds fall within precomputed max_seq_len
        return self.cos[start_pos : start_pos + seq_len], self.sin[start_pos : start_pos + seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit Feed-Forward Network."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """Multi-Head Attention block using native SDPA (FlashAttention/MemEfficient)."""
    def __init__(self, n_embed: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.block_size = block_size
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=4096)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T, start_pos=start_pos)
        q, k = apply_rope(q, k, cos, sin)

        # Cache update executes if and only if precomputed KV tensors are supplied
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        # Use native PyTorch SDPA kernel optimization
        is_causal = (T > 1 and kv_cache is None)
        if not is_causal and T > 1:
            total_seq_len = k.size(2)
            attn_mask = torch.tril(torch.ones((T, total_seq_len), device=x.device, dtype=torch.bool))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int):
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size)
        self.norm2 = RMSNorm(n_embed)
        hidden_dim = int(8 / 3 * n_embed)
        self.mlp = SwiGLU(n_embed, hidden_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class ModernLLM(nn.Module):
    """Decoder-only LLM featuring SDPA attention, KV-Cache, Top-P Nucleus Sampling, and Checkpointing."""
    def __init__(self, vocab_size: int, n_embed: int = 128, block_size: int = 64, n_layer: int = 4, n_head: int = 4):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(n_embed, n_head, block_size) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: torch.Tensor = None, 
        start_pos: int = 0, 
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ):
        x = self.tok_emb(idx)
        new_kv_caches = []

        for i, block in enumerate(self.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, start_pos=start_pos, kv_cache=layer_kv)
            new_kv_caches.append(new_kv)

        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate_kv(
        self, 
        idx: torch.Tensor, 
        max_new_tokens: int, 
        temperature: float = 0.8, 
        top_p: float = 0.9, 
        greedy: bool = False
    ):
        """Streaming token generation accelerated by Key-Value Caching and Nucleus Sampling."""
        B, T = idx.shape
        kv_caches = None
        curr_idx = idx

        for pos in range(max_new_tokens):
            if pos == 0:
                logits, _, kv_caches = self(curr_idx, start_pos=0, kv_caches=None)
            else:
                logits, _, kv_caches = self(curr_idx, start_pos=T + pos - 1, kv_caches=kv_caches)

            logits = logits[:, -1, :]
            
            if greedy or temperature <= 1e-5:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                
                # Apply Nucleus (Top-P) Filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift mask right to keep the first token exceeding threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Map filtered logits back to original order
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            curr_idx = idx_next
            yield idx_next.item()

    def save_checkpoint(self, filepath: str):
        checkpoint = {
            "model_state": self.state_dict(),
            "config": {
                "vocab_size": self.vocab_size,
                "n_embed": self.tok_emb.embedding_dim,
                "block_size": self.block_size,
                "n_layer": len(self.blocks),
                "n_head": self.blocks[0].attn.n_head
            }
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "cpu"):
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state"])
        return model
