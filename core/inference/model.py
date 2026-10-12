# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Erudite's randomly initialized RoPE/RMSNorm/SwiGLU causal decoder."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    width: int = 128
    layers: int = 4
    heads: int = 4
    context: int = 256
    def __post_init__(self):
        # Architecture accepted iff every dimensional and context invariant holds.
        integer_fields = (self.vocab_size, self.width, self.layers, self.heads, self.context)
        if not (all(type(v) is int for v in integer_fields) and self.vocab_size >= 262 and self.width > 0 and self.layers > 0 and self.heads > 0
                and self.width % self.heads == 0 and (self.width // self.heads) % 2 == 0
                and 8 <= self.context <= 32768): raise ValueError("Invalid model architecture")

class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__(); self.weight = nn.Parameter(torch.ones(width))
    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.heads = c.heads; self.dim = c.width // c.heads; self.context = c.context
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.projection = nn.Linear(c.width, c.width, bias=False)
        frequency = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        angles = torch.outer(torch.arange(c.context).float(), frequency)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)
    def rotate(self, x, offset):
        length = x.shape[2]
        cosine = torch.cat([self.cosine[offset:offset + length]] * 2, dim=-1)[None, None].to(x.dtype)
        sine = torch.cat([self.sine[offset:offset + length]] * 2, dim=-1)[None, None].to(x.dtype)
        left, right = x.chunk(2, dim=-1)
        return x * cosine + torch.cat([-right, left], dim=-1) * sine
    def forward(self, x, cache=None):
        b, t, w = x.shape
        past = 0 if cache is None else cache[0].shape[2]
        if past + t > self.context: raise ValueError("Attention context overflow")
        q, k, v = [part.reshape(b, t, self.heads, self.dim).transpose(1, 2)
                   for part in self.qkv(x).chunk(3, dim=-1)]
        q = self.rotate(q, past); k = self.rotate(k, past)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2); v = torch.cat([cache[1], v], dim=2)
        if past and t > 1:
            # Cached chunk row i may attend iff key_position <= past+i.
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (past + torch.arange(t, device=x.device)[:, None])
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=(past == 0 and t > 1))
        output = output.transpose(1, 2).contiguous().reshape(b, t, w)
        return self.projection(output), (k, v)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); hidden = ((int(8 * c.width / 3) + 7) // 8) * 8
        self.norm1 = RMSNorm(c.width); self.norm2 = RMSNorm(c.width); self.attention = Attention(c)
        self.gate = nn.Linear(c.width, hidden, bias=False)
        self.up = nn.Linear(c.width, hidden, bias=False); self.down = nn.Linear(hidden, c.width, bias=False)
    def forward(self, x, cache=None):
        attention, new_cache = self.attention(self.norm1(x), cache)
        x = x + attention; normalized = self.norm2(x)
        return x + self.down(F.silu(self.gate(normalized)) * self.up(normalized)), new_cache

class EruditeLM(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width); self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self.initialize); self.output.weight = self.embedding.weight
    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Embedding, nn.Linear)): nn.init.normal_(module.weight, std=0.02)
    def forward(self, ids, targets=None, caches=None):
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[1] < 1: raise ValueError("Expected nonempty BxT int64 token IDs")
        if caches is not None:
            if len(caches) != len(self.blocks): raise ValueError("Wrong cache layer count")
            lengths = {cache[0].shape[2] for cache in caches}
            if len(lengths) != 1: raise ValueError("Cache lengths disagree")
            for key, value in caches:
                if key.shape != value.shape or key.shape[:2] != (ids.shape[0], self.config.heads) or key.shape[-1] != self.config.width // self.config.heads:
                    raise ValueError("Invalid cache dimensions")
        x = self.embedding(ids); updated = []
        for index, block in enumerate(self.blocks):
            x, cache = block(x, None if caches is None else caches[index]); updated.append(cache)
        logits = self.output(self.norm(x)); loss = None
        if targets is not None:
            if targets.shape != ids.shape or not torch.any(targets != -100): raise ValueError("Invalid/no training targets")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-100)
        return logits, loss, updated

    @torch.inference_mode()
    def generate(self, ids, max_tokens, temperature=0.8, top_p=0.9, eos=257):
        if not (ids.ndim == 2 and ids.shape[0] == 1 and type(max_tokens) is int and max_tokens > 0 and temperature >= 0 and 0 < top_p <= 1 and eos == 257):
            raise ValueError("Invalid single-sequence generation settings")
        if ids.shape[1] + max_tokens > self.config.context: raise ValueError("Prompt plus output exceeds context")
        self.eval(); caches = None; current = ids
        for _ in range(max_tokens):
            logits, _, caches = self(current, caches=caches)
            scores = logits[:, -1, :].clone()
            # Text continuations may use byte/BPE IDs and EOS, never padding/role/BOS markers.
            scores[:, 256:262] = -float("inf"); scores[:, eos] = logits[:, -1, eos]
            if temperature == 0: token = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                sorted_scores, order = scores.sort(descending=True)
                remove = sorted_scores.softmax(-1).cumsum(-1) > top_p
                remove[:, 1:] = remove[:, :-1].clone(); remove[:, 0] = False
                scores.masked_fill_(torch.zeros_like(remove).scatter(1, order, remove), -float("inf"))
                token = torch.multinomial(scores.softmax(-1), 1)
            index = token.item()
            if index == eos: break
            yield index
            current = token
