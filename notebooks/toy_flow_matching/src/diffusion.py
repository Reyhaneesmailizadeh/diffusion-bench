"""
Flow Matching.

Forward:  z_t = (1 - t) * x + t * eps,  t in [eps_t, 1 - eps_t]
          t=0 is data, t=1 is noise.
Loss:     any (pred_type, loss_type) pair with conversions.
Sampling: Euler ODE from t=1 (noise) to t=0 (data), 50 steps.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

T_EPS = 1e-2


class FlowMatching:

    def __init__(self, sample_steps: int = 50):
        self.sample_steps = sample_steps

    def to(self, device):
        return self

    # ---- training loss -------------------------------------------------------

    def training_losses(self, model, x_0, pred_type, loss_type):
        B = x_0.shape[0]
        t = torch.rand((B, 1), device=x_0.device).clip(T_EPS, 1 - T_EPS)
        eps = torch.randn_like(x_0)
        xt = (1 - t) * x_0 + t * eps

        pred_raw = model(xt, t)

        # v = eps - x  (velocity from data toward noise)
        if loss_type == "v":
            if pred_type == "x":
                v_hat = (xt - pred_raw) / t
            elif pred_type == "eps":
                v_hat = (pred_raw - xt) / (1 - t)
            elif pred_type == "v":
                v_hat = pred_raw
            target = eps - x_0
            return F.mse_loss(v_hat, target)

        elif loss_type == "x":
            if pred_type == "x":
                x_hat = pred_raw
            elif pred_type == "eps":
                x_hat = (xt - t * pred_raw) / (1 - t)
            elif pred_type == "v":
                x_hat = xt - t * pred_raw
            return F.mse_loss(x_hat, x_0)

        elif loss_type == "eps":
            if pred_type == "x":
                eps_hat = (xt - (1 - t) * pred_raw) / t
            elif pred_type == "eps":
                eps_hat = pred_raw
            elif pred_type == "v":
                eps_hat = xt + (1 - t) * pred_raw
            return F.mse_loss(eps_hat, eps)

    # ---- sampling (Euler ODE from noise to data) -----------------------------

    @torch.no_grad()
    def sample(self, model, shape, pred_type, device="cpu"):
        x_t = torch.randn(shape, device=device)  # start at t=1 (noise)
        N = shape[0]
        ts = torch.linspace(1 - T_EPS, T_EPS, self.sample_steps + 1, device=device)

        for i in tqdm(range(self.sample_steps), desc="sampling", leave=False):
            t = ts[i].expand(N, 1)
            dt = ts[i + 1] - ts[i]  # negative (going from t=1 to t=0)
            pred_raw = model(x_t, t)

            if pred_type == "x":
                v_hat = (x_t - pred_raw) / t
            elif pred_type == "eps":
                v_hat = (pred_raw - x_t) / (1 - t)
            elif pred_type == "v":
                v_hat = pred_raw

            x_t = x_t + v_hat * dt

        return x_t
