# diffusion/beta_schedule.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    DDPM (Ho et al.) default linear beta schedule.
    For CIFAR-10 in the paper: beta_start=1e-4, beta_end=0.02, T=1000.

    Returns:
        betas: (T,) tensor in [beta_start, beta_end]
    """
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if not (0.0 < beta_start < 1.0 and 0.0 < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be in (0, 1)")
    if beta_end < beta_start:
        raise ValueError("beta_end must be >= beta_start")

    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype, device=device)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from a 1-D tensor `a` at indices `t` and reshape to broadcast over `x_shape`.

    Args:
        a: (T,) tensor
        t: (B,) int64/long tensor of timesteps in [0, T-1]
        x_shape: target shape (B, C, H, W) or similar

    Returns:
        (B, 1, 1, 1, ...) tensor broadcastable to x_shape
    """
    if a.dim() != 1:
        raise ValueError(f"`a` must be 1-D (T,), got shape {tuple(a.shape)}")
    if t.dim() != 1:
        raise ValueError(f"`t` must be 1-D (B,), got shape {tuple(t.shape)}")
    if t.dtype not in (torch.int64, torch.long):
        t = t.long()

    B = t.shape[0]
    out = a.gather(dim=0, index=t)  # (B,)
    # reshape to (B, 1, 1, 1, ...) to broadcast
    return out.view(B, *([1] * (len(x_shape) - 1)))


@dataclass
class DiffusionSchedule:
    """
    Precomputed diffusion schedule tensors for efficient training/sampling.

    All tensors are shape (T,) unless noted.
    """
    timesteps: int
    betas: torch.Tensor

    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor

    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    # Useful for sampling with epsilon-pred parameterization
    sqrt_recip_alphas: torch.Tensor                      # 1 / sqrt(alpha_t)
    sqrt_recipm1_alphas_cumprod: torch.Tensor            # sqrt(1/alphabar_t - 1)

    # Posterior q(x_{t-1} | x_t, x0) coefficients (optional but handy)
    posterior_variance: torch.Tensor                     # beta_tilde
    posterior_log_variance_clipped: torch.Tensor
    posterior_mean_coef1: torch.Tensor                   # coef on x0
    posterior_mean_coef2: torch.Tensor                   # coef on xt

    @property
    def device(self) -> torch.device:
        return self.betas.device

    @property
    def dtype(self) -> torch.dtype:
        return self.betas.dtype


def make_ddpm_schedule(
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> DiffusionSchedule:
    """
    Create DDPM linear schedule + commonly used derived terms.
    """
    betas = linear_beta_schedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        dtype=dtype,
        device=device,
    )

    # Basic terms
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=dtype, device=betas.device), alphas_cumprod[:-1]],
        dim=0,
    )

    # Forward process helpers
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Sampling helpers (epsilon-parameterization)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

    # Posterior q(x_{t-1} | x_t, x0)
    # beta_tilde = beta_t * (1 - alphabar_{t-1}) / (1 - alphabar_t)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    # log variance clipping for numerical stability (used by many implementations)
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )

    # posterior mean coefficients:
    # mean = coef1 * x0 + coef2 * xt
    posterior_mean_coef1 = (
        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    return DiffusionSchedule(
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
    )


if __name__ == "__main__":
    # Quick sanity check
    sched = make_ddpm_schedule(timesteps=1000)
    print("betas:", sched.betas.shape, sched.betas.min().item(), sched.betas.max().item())
    print("alphas_cumprod:", sched.alphas_cumprod.shape, sched.alphas_cumprod[0].item(), sched.alphas_cumprod[-1].item())
