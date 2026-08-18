"""Frozen BridgeTower teacher for cross-modal attention-alignment.

Direction: image query -> text key (`cross_modal_image_layers`) -- deliberately the same
direction as LightningDiT's own v_dit ("how much does this image patch draw on the prompt"),
because that's the direction that actually matches a "prompt adherence" goal: it's the image
reaching into the text and being shaped by it, the same mechanism cross-attention conditioning
uses in diffusion U-Nets generally (image as query pulling text content in as key/value).

Pooling statistic: entropy, not sum. BridgeTower's cross-attention is a *dedicated* module --
unlike LightningDiT's joint self-attention (where an image query's softmax competes against
image, time, AND text keys together), this softmax only ever has text as keys, so every
image-token row already sums to 1 over *just* the text keys, unconditionally. Summing that
axis (the naive analogue of a "how much of my budget goes to text" statistic) is therefore
degenerate -- it collapses to ~1.0 for every image token regardless of content, because
there's no "elsewhere" for the row to not spend on. But the *shape* of that row still varies
meaningfully: some patches attend sharply to one or two specific words (confident grounding,
low entropy), others spread attention diffusely across the whole caption (weak/no specific
grounding, high entropy) -- so entropy survives where sum doesn't.
LightningDiT._pool_image_to_text_attn computes the matching statistic on the DiT side: it
renormalizes its own text-key sub-vector (which does NOT sum to 1 on its own, since it's only
part of a wider image+time+text budget) before taking entropy, so both sides measure the same
"given engagement with text, how peaked is it" quantity, in comparable units.

HF's public API only surfaces each modality's *self*-attention via output_attentions, not
this cross-attention -- BridgeTowerBertCrossLayer computes it but the model's forward() never
collects it. This module patches that one layer to also stash it.

Verified against BridgeTower's source (transformers/models/bridgetower/modeling_bridgetower.py)
as of 2026-07, and smoke-tested end-to-end against real weights on real dataset images. Before
relying on it in a full precompute run, sanity-check that the resulting entropy map actually
looks spatially meaningful (concentrates on the described subject, not background) -- low
entropy alone doesn't prove *correct* grounding, only *confident* attention somewhere.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeTowerAttnTeacher(nn.Module):
    """Frozen teacher producing a pooled, grid-matched image->text attention-entropy target.

    Output contract matches LightningDiT._pool_image_to_text_attn's return shape: a
    [B, target_grid_size**2] vector, one non-negative entropy scalar per image token
    (lower = more confidently/peakily grounded on specific caption words, higher = more
    diffuse), already interpolated onto the DiT's own grid so no further reshaping is
    needed by the caller. See the module docstring for the direction/statistic choice.
    """

    def __init__(self, model_name="BridgeTower/bridgetower-base", device="cuda", cross_layer_depth=-1):
        super().__init__()
        from transformers import BridgeTowerModel, BridgeTowerProcessor

        self.device = device
        self.processor = BridgeTowerProcessor.from_pretrained(model_name)
        self.model = BridgeTowerModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        # Which cross_modal_image_layers block to read. BridgeTower's own depth (default
        # 6 for bridgetower-base) is independent of the DiT's attn_align_layer_depth --
        # -1 (last layer) is a reasonable default, not a matched correspondence.
        self.cross_layer_depth = cross_layer_depth

        self._captured_cross_attn = None
        self._patch_cross_layer()

    def _patch_cross_layer(self):
        """Wrap one BridgeTowerBertCrossLayer.forward to also stash its cross-attention
        weights. BridgeTowerBertCrossLayer returns
            (layer_output, self_attn_weights, cross_attn_weights)
        when called with output_attentions=True; BridgeTowerModel.forward() only keeps
        self_attn_weights (index 1) for its public `attentions` output, discarding
        cross_attn_weights (index 2) -- that's the one we want.
        """
        layer = self.model.cross_modal_image_layers[self.cross_layer_depth]
        orig_forward = layer.forward

        def patched_forward(*args, **kwargs):
            kwargs["output_attentions"] = True
            out = orig_forward(*args, **kwargs)
            self._captured_cross_attn = out[2]
            return out

        layer.forward = patched_forward

    @torch.no_grad()
    def _raw_entropy_grid(self, images, captions):
        """Run BridgeTower and return the native-resolution [B, side, side] entropy grid,
        before any interpolation to the DiT's grid. Exposed separately (not just inlined
        into get_attn_target) so a sanity-check visualization can look at this directly,
        overlaid on the source image, before trusting it as a training target.
        """
        inputs = self.processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self._captured_cross_attn = None
        self.model(**inputs, output_attentions=True)
        cross_attn = self._captured_cross_attn  # expected: [B, heads, image_tokens, text_len]
        if cross_attn is None:
            raise RuntimeError(
                "BridgeTower cross-attention patch did not fire -- check that "
                "cross_modal_image_layers[cross_layer_depth].forward still returns "
                "(layer_output, self_attn_weights, cross_attn_weights) in your transformers version."
            )

        text_attn_mask = inputs.get("attention_mask")  # [B, text_len], 1 for real (non-pad) tokens
        eps = 1e-8

        pooled = cross_attn.mean(dim=1)  # [B, image_tokens, text_len] -- average over heads
        # Each image-token row already sums to 1 over just the text keys (dedicated cross-
        # attention, nothing else to spend on) -- mask + renormalize over real tokens only,
        # defensively, in case BridgeTower's own internal masking didn't already zero padding.
        if text_attn_mask is not None:
            mask = text_attn_mask.unsqueeze(1).to(pooled.dtype)  # [B, 1, text_len]
            pooled = pooled * mask
            pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(pooled.clamp_min(eps) * pooled.clamp_min(eps).log()).sum(dim=-1)  # [B, image_tokens]

        # BridgeTower's vision tower prepends a CLS token, so image_tokens is usually
        # num_patches + 1. Drop it so the rest reshapes into a square patch grid.
        num_tokens = entropy.shape[-1]
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            entropy = entropy[:, 1:]
            num_tokens -= 1
            side = math.isqrt(num_tokens)
            assert side * side == num_tokens, (
                f"Could not reshape {entropy.shape[-1]} BridgeTower image tokens into a square "
                f"grid even after dropping one leading token -- check the actual token count "
                f"for this checkpoint/resolution."
            )
        return entropy.view(-1, side, side)

    @torch.no_grad()
    def get_attn_target(self, images, captions, target_grid_size):
        """
        Args:
            images: list of PIL.Image, one per sample (BridgeTowerProcessor's expected input).
            captions: list[str], same length as images.
            target_grid_size: side length of the DiT's own image-token grid (e.g. 16 for a
                16x16 = 256-token latent grid) -- the output is interpolated to match this.
        Returns:
            [B, target_grid_size ** 2] tensor of entropy values (see class docstring).
        """
        grid = self._raw_entropy_grid(images, captions).unsqueeze(1)  # [B, 1, side, side]
        grid = F.interpolate(grid, size=(target_grid_size, target_grid_size), mode="bicubic", align_corners=False)
        # Bicubic can overshoot near sharp transitions and go slightly negative even though
        # entropy itself is non-negative by construction -- clamp it back, since
        # attn_align_loss's "kl" mode normalizes this into a distribution and negative mass
        # there is meaningless.
        grid = grid.clamp_min(0)
        return grid.reshape(-1, target_grid_size * target_grid_size)
