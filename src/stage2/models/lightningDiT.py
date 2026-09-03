import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from .model_utils import ConditionEmbedder, GaussianFourierEmbedding, NormAttention, RMSNorm, RoPE, SwiGLUFFN


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * hidden_size * mlp_ratio))

    def forward(self, x, rope, attn_mask=None, return_weights=False):
        attn_out = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask, return_weights=return_weights)
        attn_weights = None
        if return_weights:
            attn_out, attn_weights = attn_out
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        if return_weights:
            return x, attn_weights
        return x


class ConvRepaProjector(nn.Module):
    """iREPA (arxiv 2512.10794) projector: a same-size conv over the image-token spatial grid,
    instead of a per-token MLP/Linear. Only valid on a pure image-token slice (no cls/register
    token mixed in) since the grid reshape assumes every token is a spatial position.
    """
    def __init__(self, hidden_size, z_dim, grid_size, kernel_size=3):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Conv2d(hidden_size, z_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        b, n, d = x.shape
        h = w = self.grid_size
        assert n == h * w, f"ConvRepaProjector expects a square image-token grid, got n={n} for grid_size={h}"
        x = x.reshape(b, h, w, d).permute(0, 3, 1, 2)  # [B, N, D] -> [B, D, H, W]
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(b, n, -1)  # [B, z_dim, H, W] -> [B, N, z_dim]
        return x


class LightningFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cls_dim=None):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        if cls_dim is not None:
            self.cls_linear = nn.Linear(hidden_size, cls_dim)

    def forward(self, x):
        x = self.norm_final(x)
        if hasattr(self, 'cls_linear'):
            cls_pred = self.cls_linear(x[:, 0, :])
            return self.linear(x[:, 1:, :]), cls_pred
        return self.linear(x)


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size=16,
        in_channels=768,
        patch_size=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        enable_repa=False,
        repa_layer_depth=8,
        repa_projector_type="mlp",
        repa_projector_kernel_size=3,
        z_dim=None,
        enable_reg=False,
        num_classes=1000,
        condition_type="label",
        context_dim=768,
        cond_arch=None,
        use_cfg_conds=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.repa_layer_depth = repa_layer_depth
        self.enable_reg = enable_reg
        self.use_cfg_conds = use_cfg_conds

        self.x_embedder = PatchEmbed(input_size, self.patch_size, in_channels, hidden_size)

        self.num_cond_tokens = cond_arch.num_t_tokens + cond_arch.num_c_tokens
        self.num_c_tokens = cond_arch.num_c_tokens  # text/context tokens, always the last num_c_tokens of the sequence
        self.t_embedder = GaussianFourierEmbedding(self.hidden_size, cond_arch.num_t_tokens)
        self.ctx_embedder = ConditionEmbedder(
            self.hidden_size, num_classes, context_dim, condition_type, cond_arch.num_c_tokens
        )
        if self.use_cfg_conds:
            self.num_cond_tokens += cond_arch.num_cfg_omega_tokens
            self.cfg_w_embedder = GaussianFourierEmbedding(self.hidden_size, cond_arch.num_cfg_omega_tokens)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(self.hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = LightningFinalLayer(self.hidden_size, self.patch_size, self.in_channels, cls_dim=z_dim if enable_reg else None)
        self.rope = RoPE(self.hidden_size // num_heads, self.x_embedder.num_patches, self.num_cond_tokens, extra_tokens=int(enable_reg))
        if enable_repa:
            if repa_projector_type == "conv":
                assert not enable_reg, (
                    "iREPA conv projector requires a pure image-token slice for the spatial "
                    "grid reshape; it doesn't support enable_reg (register/cls token) yet."
                )
                grid_size = int(round(self.x_embedder.num_patches ** 0.5))
                assert grid_size * grid_size == self.x_embedder.num_patches, (
                    f"iREPA conv projector requires a square image-token grid, "
                    f"got num_patches={self.x_embedder.num_patches}"
                )
                self.repa_projector = ConvRepaProjector(self.hidden_size, z_dim, grid_size, repa_projector_kernel_size)
            else:
                self.repa_projector = nn.Linear(self.hidden_size, z_dim)
        if enable_reg:
            self.cls_in_proj = nn.Linear(z_dim, self.hidden_size)
            self.cls_in_norm = RMSNorm(self.hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Patch embedders
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Condition embedders
        if hasattr(self.ctx_embedder, "mlp"):
            nn.init.normal_(self.ctx_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ctx_embedder.mlp[2].weight, std=0.02)
        if hasattr(self.ctx_embedder, "embedding_table"):
            nn.init.normal_(self.ctx_embedder.embedding_table.weight, std=0.02)

        # Timestep embedding MLP
        t_embedders = ["t_embedder", "cfg_w_embedder"]
        for t_embedder in t_embedders:
            if hasattr(self, t_embedder):
                nn.init.normal_(getattr(self, t_embedder).mlp[0].weight, std=0.02)
                nn.init.normal_(getattr(self, t_embedder).mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self.final_layer, "cls_linear"):
            nn.init.constant_(self.final_layer.cls_linear.weight, 0)
            nn.init.constant_(self.final_layer.cls_linear.bias, 0)

    def unpatchify(self, x):
        """[N, T, patch_size**2 * C] -> [N, C, H, W]"""
        h, c, p = int(x.shape[1] ** 0.5), self.in_channels, self.patch_size
        x = x.reshape(x.shape[0], h, h, p, p, c).permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, h*p, h*p)
        return x

    def _build_sequence(self, x, t, condition_kwargs):
        seq = []
        if self.enable_reg:
            seq.append(self.cls_in_norm(self.cls_in_proj(condition_kwargs["cls_t"])).unsqueeze(1))
        seq.append(self.x_embedder(x))
        seq.append(self.t_embedder(t))
        if self.use_cfg_conds:
            seq.append(self.cfg_w_embedder(condition_kwargs["omega"]))
        seq.append(self.ctx_embedder(condition_kwargs["context"]))
        seq = torch.cat(seq, dim=1)
        return seq

    def _build_attn_mask(self, seq, condition_kwargs):
        # Create multiplicative mask template
        attn_mask = torch.ones((seq.shape[0], seq.shape[1]), device=seq.device)
        cond_mask = condition_kwargs.get("attn_mask")
        if cond_mask is not None:
            attn_mask[:, -cond_mask.shape[1]:] = cond_mask
        # Convert to additive mask
        attn_mask = (1.0 - attn_mask[:, None, None, :]) * torch.finfo(seq.dtype).min
        return attn_mask

    def _pool_image_to_text_attn(self, attn_weights, s, n):
        """Collapse [B, heads, N, N] joint attention to a per-image-token text-attention entropy.

        Averages over heads, keeps only image-token rows and text-token columns (the last
        num_c_tokens of the sequence). Unlike a same-modality alignment target, the reference
        teacher's tokenizer generally won't match this model's, so token-level (column-to-
        column) comparison isn't meaningful -- instead we renormalize the text-key sub-vector
        to sum to 1 (it doesn't on its own here, since only *part* of this row's full
        image+time+text softmax budget goes to text) and take its entropy: "given this patch
        engages with the caption at all, how confidently/peakily is that engagement focused,"
        which is tokenizer-agnostic and comparable against the same statistic computed on a
        teacher model (see encoders/bridgetower_attn_teacher.py). Returns [B, n].
        """
        weights = attn_weights.mean(dim=1)  # [B, N, N], average over heads
        image_rows = weights[:, s:s + n, :]  # [B, n, N]
        text_cols = image_rows[:, :, -self.num_c_tokens:]  # [B, n, num_c_tokens]
        eps = 1e-8
        p = text_cols / text_cols.sum(dim=-1, keepdim=True).clamp_min(eps)
        return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1)  # [B, n]

    def forward(self, x, t, return_intermediate=False, return_attn_layer=None, **condition_kwargs):
        zt_intermediate = None
        attn_intermediate = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        s, n = int(self.enable_reg), self.x_embedder.num_patches
        for i, block in enumerate(self.blocks):
            want_weights = return_attn_layer is not None and (i + 1) == return_attn_layer
            if want_weights:
                x, attn_weights = block(x, self.rope, attn_mask, return_weights=True)
                attn_intermediate = self._pool_image_to_text_attn(attn_weights, s, n)
            else:
                x = block(x, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(x[:, :s + n, :])

        if self.enable_reg:
            x, cls_pred = self.final_layer(x[:, :s + n, :])
        else:
            x = self.final_layer(x[:, :n, :])
        x = self.unpatchify(x)
        if self.enable_reg:
            x = (x, cls_pred)

        if return_intermediate and return_attn_layer is not None:
            return x, zt_intermediate, attn_intermediate
        if return_intermediate:
            return x, zt_intermediate
        if return_attn_layer is not None:
            return x, attn_intermediate
        return x


class LightningDiTIG(LightningDiT):
    def __init__(self, base_model_depth=8, **kwargs):
        super().__init__(**kwargs)
        self.base_model_depth = base_model_depth

        self.base_final_layer = LightningFinalLayer(self.hidden_size, self.patch_size, self.in_channels)
        nn.init.constant_(self.base_final_layer.linear.weight, 0)
        nn.init.constant_(self.base_final_layer.linear.bias, 0)

    def forward(self, x, t, return_intermediate=False, **condition_kwargs):
        zt_intermediate = None
        x_base = None
        x = self._build_sequence(x, t, condition_kwargs)
        attn_mask = self._build_attn_mask(x, condition_kwargs)
        s, n = int(self.enable_reg), self.x_embedder.num_patches
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, attn_mask)
            if return_intermediate and (i + 1) == self.repa_layer_depth:
                zt_intermediate = self.repa_projector(x[:, :s + n, :])
            if (i + 1) == self.base_model_depth:
                x_base = x[:, s:s + n, :]

        if self.enable_reg:
            x, cls_pred = self.final_layer(x[:, :s + n, :])
        else:
            x = self.final_layer(x[:, :n, :])
        x = self.unpatchify(x)
        x_base = self.unpatchify(self.base_final_layer(x_base))

        out = (x, x_base, cls_pred) if self.enable_reg else (x, x_base)
        if return_intermediate:
            return out, zt_intermediate
        return out
