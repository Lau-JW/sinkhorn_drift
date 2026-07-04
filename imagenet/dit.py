"""DiT with LightningDiT architecture fully aligned with Drifting.

Components ported from drifting's models/generator.py:
  - LightningDiTBlock: AdaLN + RMSNorm + QK-Norm + RoPE
  - SwiGLU FFN (replaces standard GELU MLP)
  - RoPE rotary positional embeddings (replaces learned pos_embed)
  - n_cls_tokens: class tokens prepended to sequence
  - FinalLayer with AdaLN modulation
  - Zero init on all adaLN outputs and final linear

Plus sinkhorn-specific improvements:
  - noise_embeds (noise_classes × noise_coords)
  - cfg_embedder + cfg_norm for drifting-aligned conditioning
  - Both "drift" and "flow" (backward compat) conditioning modes
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility Functions ─────────────────────────────────────────────────────────

def _sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP -> hidden_size."""
    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_dim = frequency_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(_sinusoidal_embedding(t, self.frequency_dim))


class LabelEmbedder(nn.Module):
    """Simple class embedding (no dropout — CFG handled via cfg_scale)."""
    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class RMSNorm(nn.Module):
    """RMSNorm — used in drifting's LightningDiT."""
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        if self.elementwise_affine and self.weight is not None:
            normed = normed * self.weight.to(x.dtype)
        return normed.to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting over token dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """2D image → patch tokens (same as drifting's TorchLinear after reshape)."""
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, hidden, H/p, W/p]
        return x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]


# ── RoPE ─────────────────────────────────────────────────────────────────────

def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary Positional Embedding on q, k.

    Args:
        q, k: [B, N, H, D] (batch, seq_len, heads, head_dim)
    Returns:
        q_embed, k_embed: [B, N, H, D]
    """
    B, N, H, D = q.shape
    half_dim = D // 2
    freqs = (1.0 / (10000.0 ** (torch.arange(0, half_dim, device=q.device, dtype=torch.float32) / half_dim)))
    t = torch.arange(N, device=q.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [N, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [N, D]

    cos = emb.cos()[None, :, None, :]  # [1, N, 1, D]
    sin = emb.sin()[None, :, None, :]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN from drifting: SiLU(w1(x)) * w3(x) → Linear."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ── Attention with QK-Norm + RoPE ────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head self-attention with optional QK-Norm and RoPE."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

        if qk_norm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
                self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            q, k = apply_rope(q, k)

        # [B, H, N, D] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn)


# ── LightningDiT Block ───────────────────────────────────────────────────────

class LightningDiTBlock(nn.Module):
    """Drifting-aligned DiT block: AdaLN + RMSNorm + Attention(QK-Norm+RoPE) + SwiGLU."""
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm

        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        norm_kw = {"eps": 1e-6} if use_rmsnorm else {"eps": 1e-6, "elementwise_affine": False}
        self.norm1 = norm_cls(hidden_size, **norm_kw)
        self.norm2 = norm_cls(hidden_size, **norm_kw)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = (int(2 / 3 * mlp_hidden) + 31) // 32 * 32  # round to multiple of 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_hidden, hidden_size),
            )

        # AdaLN modulation: cond → 6 * hidden_size (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )

        # Attention with AdaLN
        x_norm = self.norm1(x)
        x_norm = modulate(x_norm, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)

        # MLP with AdaLN
        x_norm = self.norm2(x)
        x_norm = modulate(x_norm, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """Drifting-aligned final layer: norm → AdaLN → linear → unpatchify."""
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        norm_kw = {"eps": 1e-6} if use_rmsnorm else {"eps": 1e-6, "elementwise_affine": False}
        self.norm_final = norm_cls(hidden_size, **norm_kw)

        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Zero init
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ── DiTConfig ────────────────────────────────────────────────────────────────

@dataclass
class DiTConfig:
    in_channels: int = 4
    out_channels: int = 4
    input_size: int = 32
    patch_size: int = 2
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    noise_classes: int = 64
    noise_coords: int = 32
    n_cls_tokens: int = 16          # drifting-aligned
    use_qknorm: bool = True
    use_swiglu: bool = True
    use_rope: bool = True
    use_rmsnorm: bool = True
    learn_sigma: bool = False

    @property
    def output_channels(self) -> int:
        return self.out_channels * 2 if self.learn_sigma else self.out_channels


class DiT(nn.Module):
    """LightningDiT (drifting-aligned architecture).

    Supports two conditioning modes:
      - "drift":  cond = y_embed(y) + Σnoise_embed_i(nl_i) + cfg_norm(cfg_embed(cfg))
      - "flow":   cond = t_embed(t) + h_embed(h) + y_embed(y)

    Flow mode is selected by omitting cfg_scale/noise_labels.
    """

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        self.out_channels = config.output_channels
        self.patch_embed = PatchEmbed(config.in_channels, config.hidden_size, config.patch_size)
        num_patches = (config.input_size // config.patch_size) ** 2

        # Drifting-aligned: 2D sincos positional embedding (not learned)
        self.register_buffer("pos_embed", self._build_sincos_pos_embed(config.hidden_size, num_patches))

        # Flow-mode embedders (backward compat)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.h_embedder = TimestepEmbedder(config.hidden_size)

        # Class embedder
        self.y_embedder = LabelEmbedder(config.num_classes, config.hidden_size)

        # Drift-mode: noise embeddings
        self.noise_classes = config.noise_classes
        self.noise_coords = config.noise_coords
        if self.noise_classes > 0:
            self.noise_embeds = nn.ModuleList([
                nn.Embedding(self.noise_classes, config.hidden_size)
                for _ in range(self.noise_coords)
            ])

        # Drift-mode: CFG embedder
        self.cfg_embedder = TimestepEmbedder(config.hidden_size)
        self.cfg_norm = RMSNorm(config.hidden_size)

        # Class tokens
        self.n_cls_tokens = config.n_cls_tokens
        if self.n_cls_tokens > 0:
            self.cls_embed = nn.Parameter(torch.randn(1, self.n_cls_tokens, config.hidden_size) * 0.02)
            self.cls_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

        # DiT blocks
        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                use_qknorm=config.use_qknorm,
                use_swiglu=config.use_swiglu,
                use_rmsnorm=config.use_rmsnorm,
                use_rope=config.use_rope,
            ) for _ in range(config.depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=self.out_channels,
            use_rmsnorm=config.use_rmsnorm,
        )

        self._init_weights()

    def _build_sincos_pos_embed(self, dim: int, num_patches: int) -> torch.Tensor:
        """2D sinusoidal pos embed (same as drifting)."""
        grid_size = int(math.sqrt(num_patches))
        assert grid_size * grid_size == num_patches

        def _1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
            assert embed_dim % 2 == 0
            half = embed_dim // 2
            omega = torch.arange(half, dtype=torch.float32) / half
            omega = 1.0 / (10000.0 ** omega)
            pos = pos.reshape(-1)
            out = torch.einsum("m,d->md", pos, omega)
            emb_sin = torch.sin(out)
            emb_cos = torch.cos(out)
            return torch.cat([emb_sin, emb_cos], dim=1)

        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # w first
        grid = torch.stack(grid, dim=0).reshape(2, -1)

        emb_h = _1d_sincos(dim // 2, grid[0])
        emb_w = _1d_sincos(dim // 2, grid[1])
        pe = torch.cat([emb_h, emb_w], dim=1)  # [num_patches, dim]
        return pe[None, :, :]  # [1, num_patches, dim]

    def _init_weights(self) -> None:
        # Zero init all adaLN modulations
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Final layer already zero-init in __init__

        # Zero init cfg_embedder output
        nn.init.constant_(self.cfg_norm.weight, 1)
        nn.init.constant_(self.cfg_embedder.mlp[-1].weight, 0)
        nn.init.constant_(self.cfg_embedder.mlp[-1].bias, 0)

        # Normal init for noise embeddings
        for embed in getattr(self, "noise_embeds", []):
            nn.init.normal_(embed.weight, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.config.patch_size
        h = w = self.config.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        *,
        train: bool = False,
        force_drop_ids: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        noise_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate output.

        Args:
            x: input noise [B, C, H, W]
            t: timestep [B] (unused in drift mode)
            y: class labels [B]
            h: flow interval [B] (flow mode only)
            cfg_scale: CFG strength [B] (drift mode). When provided, use drift conditioning.
            noise_labels: [B, noise_coords] (drift mode). When provided, use drift conditioning.
        """
        # Patch embed
        x = self.patch_embed(x)  # [B, num_patches, hidden]

        # Add positional embedding
        x = x + self.pos_embed

        # Build conditioning
        if cfg_scale is not None and noise_labels is not None:
            # ── Drift-mode conditioning (aligned with drifting's DitGen) ──
            cond = self.y_embedder(y)
            for i, embed in enumerate(self.noise_embeds):
                cond = cond + embed(noise_labels[:, i])
            cfg_emb = self.cfg_embedder(cfg_scale)
            cond = cond + self.cfg_norm(cfg_emb) * 0.02
        else:
            # ── Flow-mode conditioning (backward compat) ──
            if h is None:
                h = torch.zeros_like(t, dtype=t.dtype, device=t.device)
            cond = self.t_embedder(t) + self.h_embedder(h) + self.y_embedder(y)

        # Add class tokens (drifting-aligned)
        if self.n_cls_tokens > 0:
            if cfg_scale is not None:
                # Use class embedding as class token source
                c_tokens = self.cls_proj(cond).unsqueeze(1)  # [B, 1, hidden]
                c_tokens = c_tokens.expand(-1, self.n_cls_tokens, -1)  # [B, n_cls, hidden]
                c_tokens = c_tokens + self.cls_embed  # add learnable cls_embed
                x = torch.cat([c_tokens, x], dim=1)  # [B, n_cls + num_patches, hidden]

        if extra_cond is not None:
            cond = cond + extra_cond

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final layer
        x = self.final_layer(x, cond)

        # Remove class tokens
        if self.n_cls_tokens > 0 and cfg_scale is not None and noise_labels is not None:
            x = x[:, self.n_cls_tokens:]

        # Unpatchify
        return self.unpatchify(x)


# ── Factory functions ────────────────────────────────────────────────────────

def _make_dit_config(name: str, **overrides) -> DiTConfig:
    presets = {
        "S/2": dict(hidden_size=384, depth=12, num_heads=6, patch_size=2),
        "B/4": dict(hidden_size=768, depth=12, num_heads=12, patch_size=4),
        "B/2": dict(hidden_size=768, depth=12, num_heads=12, patch_size=2),
        "L/2": dict(hidden_size=1024, depth=24, num_heads=16, patch_size=2),
        "XL/2": dict(hidden_size=1152, depth=28, num_heads=16, patch_size=2),
    }
    if name not in presets:
        raise ValueError(f"Unknown DiT preset {name!r}, choose from {list(presets)}")
    cfg = DiTConfig(**presets[name])
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def dit_s_2(**kwargs) -> DiT:
    return DiT(_make_dit_config("S/2", **kwargs))


def dit_b_4(**kwargs) -> DiT:
    return DiT(_make_dit_config("B/4", **kwargs))


def dit_b_2(**kwargs) -> DiT:
    return DiT(_make_dit_config("B/2", **kwargs))


def dit_l_2(**kwargs) -> DiT:
    return DiT(_make_dit_config("L/2", **kwargs))


def dit_xl_2(**kwargs) -> DiT:
    return DiT(_make_dit_config("XL/2", **kwargs))
