"""CPU implementations of the paper's ViT and GPT-2 FFN comparisons."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


PAIRS = (("relu", "reglu"), ("gelu", "geglu"), ("silu", "swiglu"))


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name in ("relu", "reglu"):
        return F.relu(x)
    if name in ("gelu", "geglu"):
        return F.gelu(x)
    if name in ("silu", "swiglu"):
        return F.silu(x)
    raise ValueError(name)


class MatchedFFN(nn.Module):
    """Two-layer FFN; gated width is reduced to match parameter count."""

    def __init__(self, dim: int, hidden: int, activation: str, seed: int):
        super().__init__()
        self.activation = activation
        self.gated = activation in {"reglu", "geglu", "swiglu"}
        if self.gated:
            hidden = int(hidden * (dim + dim) / (2 * dim + dim))
            self.value = nn.Linear(dim, hidden)
            self.gate = nn.Linear(dim, hidden)
        else:
            self.value = nn.Linear(dim, hidden)
            self.gate = None
        self.out = nn.Linear(hidden, dim)
        self.reset(seed)

    def reset(self, seed: int) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for layer in (self.value, self.gate, self.out):
                if layer is not None:
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        if self.gate is None:
            hidden = _activation(self.activation, value)
        else:
            hidden = value * _activation(self.activation, self.gate(x))
        return self.out(hidden)


class Attention(nn.Module):
    """Multi-head attention with the paper code's configurable inner dimension."""

    def __init__(self, dim: int, heads: int, head_dim: int, seed: int, causal: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        inner = heads * head_dim
        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.out = nn.Linear(inner, dim)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.qkv.weight)
            nn.init.xavier_uniform_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(
            batch, length, self.heads * self.head_dim
        )
        return self.out(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        hidden: int,
        activation: str,
        seed: int,
        *,
        causal: bool,
        head_dim: int,
    ):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, head_dim, seed, causal)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MatchedFFN(dim, hidden, activation, seed + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attended = self.attention(normed)
        x = x + attended
        return x + self.ffn(self.norm2(x))


class PaperViT(nn.Module):
    """Official repository Figure-3 ViT configuration."""

    def __init__(self, activation: str, seed: int):
        super().__init__()
        image_size, patch, dim, depth, heads, hidden = 32, 4, 4096, 4, 4, 256
        patch_dim = 3 * patch * patch
        self.patch = patch
        self.patch_projection = nn.Linear(patch_dim, dim)
        with torch.random.fork_rng():
            torch.manual_seed(seed + 700)
            nn.init.xavier_uniform_(self.patch_projection.weight)
            nn.init.zeros_(self.patch_projection.bias)
            self.cls = nn.Parameter(torch.randn(1, 1, dim))
            self.position = nn.Parameter(
                torch.randn(1, (image_size // patch) ** 2 + 1, dim)
            )
            readout = torch.randn(dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    hidden,
                    activation,
                    seed + 1000 + 10 * layer,
                    causal=False,
                    head_dim=64,
                )
                for layer in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("readout", readout / readout.norm())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        p = self.patch
        patches = image.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(image.shape[0], -1, 3 * p * p)
        x = self.patch_projection(patches)
        cls = self.cls.expand(image.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.position
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0] @ self.readout


class GPT2SmallBackbone(nn.Module):
    """Standard GPT-2-small block dimensions, operating on input embeddings."""

    def __init__(self, activation: str, seed: int):
        super().__init__()
        dim, depth, heads, hidden, sequence = 768, 12, 12, 3072, 16
        with torch.random.fork_rng():
            torch.manual_seed(seed + 800)
            self.position = nn.Parameter(torch.randn(1, sequence, dim) * 0.02)
            readout = torch.randn(dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    hidden,
                    activation,
                    seed + 2000 + 10 * layer,
                    causal=True,
                    head_dim=64,
                )
                for layer in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("readout", readout / readout.norm())

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = embeddings + self.position
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, -1] @ self.readout


def ffn_parameters(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if ".ffn." in name:
            parameter.requires_grad_(True)
            selected.append(parameter)
    return selected


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in ffn_parameters(model))
