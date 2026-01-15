from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

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
    if timesteps <= 0:
        raise ValueError("timesteps must be greater than 0")
    if not(0.0 < beta_start < beta_end < 1.0):
        raise ValueError("beta_start and beta_end must be between 0 and 1")
    
    return torch.linspace(beta_start, beta_end, timesteps, dtype=dtype, device=device)

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    미리 계산된 전체 확산 계수 리스트(a)에서, 현재 batch(t)에 해당하는 값만 뽑아
    이미지 텐서(x_shape)와 연산할 수 있도록 차원을 맞춰주는 함수입니다.

    Args:
        a (torch.Tensor): 전체 타임스텝에 대한 확산 계수 테이블 (Lookup Table)
            - 역할: 전체 스케줄 정보 (예: 0~1000까지의 모든 sqrt_alpha_cumprod 값)
            - 형태: (Timesteps,) 예: (1000,)

        t (torch.Tensor): 현재 처리 중인 미니배치의 타임스텝 인덱스 (Query Indices)
            - 역할: 각 이미지가 확산 과정의 몇 번째 단계에 있는지 나타냄
            - 형태: (Batch_size,) 예: (32,) -> [50, 999, 12, ...]

        x_shape (Tuple[int, ...]): 대상이 되는 이미지 데이터의 형태
            - 역할: 리턴값을 이 모양에 맞춰 브로드캐스팅하기 위해 참조함
            - 형태: (Batch_size, Channel, Height, Width) 예: (32, 3, 32, 32)

    Returns:
        torch.Tensor: 이미지 텐서와 바로 곱하기/더하기가 가능한 형태의 텐서
            - 형태: (Batch_size, 1, 1, 1) -> 뒤쪽 차원이 1로 채워짐
    """
    if a.dim() != 1:
        raise ValueError(f"a must be 1-D, got shape {a.shape}")
    if t.dim() != 1:
        raise ValueError(f"t must be 1-D, got shape {t.shape}")
    if t.dtype not in (torch.int64, torch.long):
        t = t.long()
    
    B = t.shape[0]
    # 1. Gather : 전체 계수 'a'에서, 우리에게 필요한 시점 't'의 값만 뽑아옴
    # 결과 형태: (Batch_size,)
    out = a.gather(dim=0, index=t)

    # 2. Reshape for Broadcasting:
    # 이미지 (B,C,H,W)와 연산을 할려면(곱할려면), 계수 텐서도 차원 수가 같아야 한다.
    # 따라서 (Batch_size,) -> (Batch_size, 1, 1, 1)로 변환(나머지 뒤쪽을 1로 채운다)
    # 이렇게 하면 PyTorch가 알아서 나머지 공간 (C,H,W)에 값을 복사해서 연산.
    return out.view(B, *([1] * (len(x_shape) - 1)))

@dataclass
class DiffusionSchedule:
    