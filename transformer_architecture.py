from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 8000
    context_length: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    layer_norm_eps: float = 1e-5


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.output = nn.Linear(config.d_model, config.d_model, bias=True)
        self.register_buffer(
            "causal_mask",
            torch.ones(config.context_length, config.context_length,
                       dtype=torch.bool).tril(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.n_heads, self.head_dim)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5
        scores = scores.masked_fill(~self.causal_mask[:length, :length],
                                    float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attended = (weights @ v).transpose(1, 2).contiguous()
        return self.output(attended.reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_attention(x))
        return x + self.mlp(self.ln_mlp(x))


class SelectiveTransformer(nn.Module):
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        selected_token_ids: Optional[Sequence[int]] = None,
        eligible_token_count: Optional[int] = None,
    ):
        super().__init__()
        self.config = config or TransformerConfig()
        c = self.config
        if min(c.vocab_size, c.context_length, c.n_layers,
               c.d_model, c.n_heads, c.d_ff) <= 0:
            raise ValueError("All architecture dimensions must be positive.")
        if c.d_model % c.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        ids = [] if selected_token_ids is None else list(selected_token_ids)
        if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
            raise TypeError("selected_token_ids must contain Python integers.")
        if len(set(ids)) != len(ids):
            raise ValueError("Selected token IDs must be unique.")
        if any(i < 0 or i >= c.vocab_size for i in ids):
            raise ValueError("Selected token ID outside vocabulary.")
        if len(ids) > 400:
            raise ValueError("The proposal permits at most 400 selected rows.")
        if ids:
            if (not isinstance(eligible_token_count, int)
                    or isinstance(eligible_token_count, bool)
                    or not 0 <= eligible_token_count <= c.vocab_size):
                raise ValueError("Supply eligible_token_count to enforce the 10% cap.")
            if len(ids) > eligible_token_count // 10:
                raise ValueError("Selected rows exceed 10% of the eligible pool.")
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.position_embedding = nn.Embedding(c.context_length, c.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.d_model, eps=c.layer_norm_eps)
        self.register_buffer("selected_token_ids", torch.tensor(ids, dtype=torch.long))
        self.output_offsets = nn.Parameter(torch.zeros(len(ids), c.d_model))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return unnormalized next-token logits [batch, length, 8000]."""
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long with shape [batch, length].")
        length = input_ids.shape[1]
        if not 1 <= length <= self.config.context_length:
            raise ValueError(f"Sequence length must be 1..{self.config.context_length}.")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        hidden = self.final_norm(x)
        # F.linear uses the SAME Parameter as input lookup: genuine weight tying.
        logits = F.linear(hidden, self.token_embedding.weight)
        if self.selected_token_ids.numel():
            corrections = F.linear(hidden, self.output_offsets)
            logits = logits.index_add(-1, self.selected_token_ids, corrections)
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # Random initialization and one dummy forward pass only; no training.
    model = SelectiveTransformer()
    model.eval()
    dummy_ids = torch.randint(0, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(dummy_ids)
    print(model)
    print(f"Unique parameters: {model.parameter_count():,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    # Optional extension, once selected IDs are available from calibration:
    # model = SelectiveTransformer(selected_token_ids=selected_ids,
    #                              eligible_token_count=eligible_count)
