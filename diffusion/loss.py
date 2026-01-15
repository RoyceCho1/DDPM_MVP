from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from diffusion.forward import q_sample
# from diffusion.beta_schedule import DiffusionSchedule

def p_losses(
    denoise_model: torch.nn.Module,
    x_start: torch.Tensor,
    t: torch.Tensor,
    schedule, # Type: DiffusionSchedule
    noise: Optional[torch.Tensor] = None,
    loss_type: str = "l2"
) -> torch.Tensor:
    """
    DDPM Training Loss Calculation (Eq. 14 in paper)
    
    L_simple = MSE(epsilon, epsilon_theta(x_t, t))

    Args:
        denoise_model (torch.nn.Module): 노이즈를 예측하는 딥러닝 모델 (U-Net)
                                       Input: (x_t, t) -> Output: predicted_noise
        x_start (torch.Tensor): 원본 이미지 (Batch, Channel, Height, Width)
        t (torch.Tensor): timestep (Batch,)
        schedule (DiffusionSchedule): 스케줄 객체
        noise (torch.Tensor, optional): 정답 노이즈 (epsilon). None이면 내부 생성.
        loss_type (str): "l1" 또는 "l2" (MSE)

    Returns:
        torch.Tensor: Scalar loss value
    """

    # 0. Noise Generation(epsilon)
    # 논문에서 정규분포 N(0, I)에서 샘플링
    if noise is None:
        noise = torch.randn_like(x_start)

    # 1. Forward Process: 이미지에 노이즈 주입 (x_0 -> x_t)
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    # q_sample 함수가 sqrt(alpha_bar_t)와 sqrt(1 - alpha_bar_t)를 계산하여 이미지에 노이즈를 주입
    x_noisy = q_sample(x_start=x_start, t=t, schedule=schedule, noise=noise)

    # 2. Predict Noise: 모델이 노이즈를 예측
    # epsilon_theta = model(x_t, t)
    # 모델에게 "x_t와 t를 줄테니깐, 여기에 들어간 노이즈를 맞춰봐"
    predicted_noise = denoise_model(x_noisy, t)

    # 3. Calculate Loss
    # 정답 noise와 예측 noise의 차이를 계산
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss
