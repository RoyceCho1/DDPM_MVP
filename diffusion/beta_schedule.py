from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
from diffusion.utils import extract

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    DDPM linear beta schedule.
    For CIFAR-10 in the paper, beta_start = 0.0001, beta_end = 0.02, T = 1000.
    beta 정의: 노이즈 분산
    """

    # 예외 처리
    if timesteps <= 0:
        raise ValueError("timesteps must be greater than 0")
    if not(0.0 < beta_start < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be between 0 and 1")
    
    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype, device=device)


@dataclass
class DiffusionSchedule:
    """
    Diffusion Process에 필요한 모든 미리 계산된 값들을 저장하는 클래스
    이를 통해 모델이나 샘플링 함수에 개별 텐서들을 일일이 넘겨주지 않고, 이 객체 하나로 깔끔하게 관리할 수 있다.

    모든 텐서는 기본적으로 (T,) 형태의 텐서로 저장된다.(timesteps 개수만큼의 길이)
    """
    timesteps: int # 타임스텝 수
    betas: torch.Tensor #노이즈 스케줄 (beta_t)

    alphas: torch.Tensor # 1-beta_t
    alphas_cumprod: torch.Tensor # alpha_t의 누적 곱 (alpha_bar_t)
    alphas_cumprod_prev: torch.Tensor # alpha_t-1의 누적 곱 (alpha_bar_t-1)

    # Forward Process q(x_t | x_0)를 위한 계수(Eq.4)
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    sqrt_alphas_cumprod: torch.Tensor # sqrt(alpha_bar_t)
    sqrt_one_minus_alphas_cumprod: torch.Tensor # sqrt(1 - alpha_bar_t)

    # Sampling을 위한 보조 계수들 (Algorithm 2)
    sqrt_recip_alphas: torch.Tensor # 1 / sqrt(alpha_t)
    sqrt_recipm1_alphas_cumprod: torch.Tensor # sqrt(1 / alpha_bar_t - 1)

    # Posterior q(x_{t-1} | x_t, x_0) 분포를 위한 계수들 (Eq. 7, Eq. 6)
    # 이 계수들은 x_0를 알 때 x_{t-1}을 추정하는 "정답 경로" 계산에 사용.
    posterior_variance: torch.Tensor # beta_tilde_t (사후 확률 분산)
    posterior_log_variance_clipped: torch.Tensor # log(beta_tilde_t), 수치 안정화를 위한 클리핑
    posterior_mean_coef1: torch.Tensor # 사후 확률 평균 계산 시 x_0 앞에 붙는 계수
    posterior_mean_coef2: torch.Tensor # 사후 확률 평균 계산 시 x_t 앞에 붙는 계수

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
    DDPM 논문의 설정에 따라 Linear Beta Schedule을 생성하고,
    이에 파생되는 모든 Diffusion Parameters(학습되지 않는)를 미리 계산해여 DiffusionSchedule 객체로 반환.
    """

    # 1. Beta Schedule 생성
    # 논문 Section 4 실험 설정 : 0.0004에서 0.02까지의 Beta Schedule(T=1000)
    betas = linear_beta_schedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        dtype=dtype,
        device=device,
    )
    
    # 2. 기본 계수 계산 (Basic Terms)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=dtype, device=device), alphas_cumprod[:-1]],
        dim=0,
    ) # alpha_bar_{t-1} (posterior 계산용). t=0일 때를 위해 1.0을 맨 앞에 추가

    # 3. Forward Process q(x_t | x_0)를 위한 계수(Eq.4)
    # q(x_t | x_0) 분포의 평균과 분산 계수
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # x_0 계수
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # epsilon 계수

    # 4. Sampling을 위한 보조 계수들 (Algorithm 2)
    
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # x_{t-1}을 복원할 때 전체 스케일을 맞춰주는 계수
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0) # x_t에서 x_0를 복원할 때 계수
    
    # 5. Posterior q(x_{t-1} | x_t, x_0) 분포를 위한 계수들 (Eq. 7)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    ) # 수치 안정화를 위한 클리핑, t=0일 때 posterior_variance가 0이 되는 것을 방지(하한은 1e-20)
    
    # q(x_{t-1} | x_t, x_0)의 평균(mean)인 mu_tilde_t 계산을 위한 계수들 (Eq. 7)
    # mu_tilde_t = coef1 * x_0 + coef2 * x_t
    posterior_mean_coef1 = (
        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
    )
    
    return DiffusionSchedule(
        timesteps = timesteps,
        betas = betas,
        alphas = alphas,
        alphas_cumprod = alphas_cumprod,
        alphas_cumprod_prev = alphas_cumprod_prev,
        sqrt_alphas_cumprod = sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas = sqrt_recip_alphas,
        sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod,
        posterior_variance = posterior_variance,
        posterior_log_variance_clipped = posterior_log_variance_clipped,
        posterior_mean_coef1 = posterior_mean_coef1,
        posterior_mean_coef2 = posterior_mean_coef2,
    )

if __name__ == "__main__":
    sched = make_ddpm_schedule(timesteps=1000)
    print("betas:", sched.betas.shape, sched.betas.min().item(), sched.betas.max().item())
    print("alphas_cumprod:", sched.alphas_cumprod.shape, sched.alphas_cumprod[0].item(), sched.alphas_cumprod[-1].item())
    