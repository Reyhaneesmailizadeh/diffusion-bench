"""
MeanFlow — one-step generation via flow matching.

Minimal adaptation of https://github.com/Gsunshine/meanflow/blob/main/meanflow.py
for toy data. Removed: guidance, class conditioning, logit-normal schedule.

Convention: t=0 data, t=1 noise.  z_t = (1-t)*x + t*eps.
Model predicts x (clean data). Converted to velocity: u = (z - x_pred) / clip(t).
Only x-prediction is supported in this file.
"""

import torch

T_EPS = 1e-3
T_CLIP_MIN = 0.05  # clip t to avoid division by zero in x-to-u conversion


class MeanFlow:
    """
    Follows the original MeanFlow (https://github.com/Gsunshine/meanflow)
    with guidance, class conditioning, and logit-normal schedule removed.
    Only x-prediction is supported.
    """

    def __init__(self, fm_ratio: float = 0.50,
                 norm_p: float = 1.0, norm_eps: float = 0.01):
        self.fm_ratio = fm_ratio
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def to(self, device):
        return self

    def sample_tr(self, B, device):
        """Sample (t, r) pair. Matches original MeanFlow.sample_tr."""
        t = torch.rand((B, 1), device=device)
        r = torch.rand((B, 1), device=device)
        t, r = torch.max(t, r), torch.min(t, r)

        data_size = int(B * self.fm_ratio)
        mask = torch.arange(B, device=device) < data_size
        r = torch.where(mask.unsqueeze(1), t, r)

        return t, r

    def _x_to_u(self, z, t, x_pred):
        """Convert x-prediction to velocity: u = (z - x_pred) / clip(t)."""
        return (z - x_pred) / t.clip(min=T_CLIP_MIN)

    def training_losses(self, model, x_0):
        B = x_0.shape[0]
        device = x_0.device

        t, r = self.sample_tr(B, device)

        e = torch.randn_like(x_0)
        z_t = (1 - t) * x_0 + t * e
        v = e - x_0

        # Model predicts x, convert to velocity u for JVP
        def u_fn(z, t_in, r_in):
            x_pred = model(z, t_in, t_in - r_in)
            return self._x_to_u(z, t_in, x_pred)

        (u, du_dt) = torch.func.jvp(
            u_fn, (z_t, t, r),
            (v, torch.ones_like(t), torch.zeros_like(r)),
        )

        # Target (matches original)
        u_tgt = v - torch.clip(t - r, min=0.0, max=1.0) * du_dt
        u_tgt = u_tgt.detach()

        # Per-sample loss with adaptive weighting (matches original)
        error = (u - u_tgt) ** 2
        per_sample = error.sum(dim=-1)

        adp_wt = (per_sample + self.norm_eps) ** self.norm_p
        loss = (per_sample / adp_wt.detach()).mean()

        # Monitoring (all detached — must not affect backward graph)
        with torch.no_grad():
            h = t - r
            per_dim_err = error.detach().mean(dim=-1)
            fm_w = (h.squeeze(-1) < 1e-6).float()
            mf_w = 1.0 - fm_w
            mse_fm = (per_dim_err * fm_w).sum() / fm_w.sum().clamp(min=1)
            mse_mf = (per_dim_err * mf_w).sum() / mf_w.sum().clamp(min=1)
            dudt_norm = (du_dt.detach() ** 2).mean().sqrt()

        return loss, mse_fm, mse_mf, dudt_norm

    @torch.no_grad()
    def sample(self, model, shape, device="cpu", num_steps=1):
        """Sample from noise (t=1) to data (t=0). Model predicts x."""
        z = torch.randn(shape, device=device)
        N = shape[0]

        ts = torch.linspace(1, 0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = ts[i].expand(N, 1)
            r = ts[i + 1].expand(N, 1)
            x_pred = model(z, t, t - r)
            u = self._x_to_u(z, t, x_pred)
            z = z - (t - r) * u

        return z
