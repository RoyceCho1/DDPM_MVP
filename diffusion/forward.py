from __future__ import annotations
from typing import Optional
import torch

from diffusion.utils import extract


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    schedule, # Type: DiffusionSchedule
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Diffuse the data (t steps)
    Forward Diffusion Process: q(x_t | x_0) (Eq. 4)
    x_t = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * epsilon

    Args:
        x_start (torch.Tensor): 원본 이미지 (Batch, Channel, Height, Width), 범위 [-1, 1]
        t (torch.Tensor): timestep (Batch,)
        schedule (DiffusionSchedule): 미리 계산된 스케줄 객체
        noise (torch.Tensor, optional): 주입할 가우시안 노이즈 (epsilon). 
                                      None이면 내부에서 생성.

    Returns:
        torch.Tensor: 노이즈가 섞인 이미지 x_t
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    # 1. x_0의 계수: sqrt(alpha_bar_t)
    # schedule.sqrt_alphas_cumprod에서 t번째 값을 추출하여 (B, 1, 1, 1)로 변환
    sqrt_alphas_cumprod_t = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape)

    # 2. epsilon의 계수: sqrt(1 - alpha_bar_t)
    # schedule.sqrt_one_minus_alphas_cumprod에서 t번째 값을 추출하여 (B, 1, 1, 1)로 변환
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    # 3. Reparameterization Trick 적용
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
