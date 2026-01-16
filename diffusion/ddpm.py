from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from diffusion.beta_schedule import make_ddpm_schedule
from diffusion.loss import p_losses
from diffusion.sampling import p_sample_loop

class DDPM(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Models) Wrapper Class
    
    이 클래스는 다음 구성요소들을 통합 관리합니다:
    1. Denoise Model (U-Net) - 학습 대상
    2. Diffusion Schedule - 미리 계산된 diffusion parameters
    3. Forward Process (Loss Calculation) - 학습 시 loss 계산
    4. Reverse Process (Sampling) - sample 생성
    """
    
    def __init__(
        self,
        denoise_model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss_type: str = 'l2'
    ):
        super().__init__()
        self.model = denoise_model
        self.loss_type = loss_type
        

        # 1. Diffusion Schedule 생성
        _schedule = make_ddpm_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=torch.device('cpu')
        )
        # 2. Schedule의 모든 tensor를 buffer로 등록
        # 이렇게 해야 ddpm.cuda()를 했을 때 스케줄 값들도 자동으로 GPU로 이동
        self.register_buffer('timesteps', torch.tensor(_schedule.timesteps))

        for key, value in _schedule.__dict__.items():
            if torch.is_tensor(value):
                self.register_buffer(key, value)

        # 구조 복원용 껍데기 저장
        self._schedule_struct = _schedule

    @property
    def schedule(self) -> DiffusionSchedule:    
        """
        현재 디바이스에 위치한 버퍼들을 모아서 DiffusionSchedule 객체로 반환
        """
        current_schedule = self._schedule_struct
        #등록된 버퍼(self.betas 등등)의 현재 값을 schedule 객체에 덮어씌움
        for key in current_schedule.__dict__.keys():
            if hasattr(self, key):
                setattr(current_schedule, key, getattr(self, key))
        return current_schedule
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training Step (Algorithm 1)
        x: (Batch, Channel, Height, Width) - 원본 이미지 (Normalized to [-1, 1])
        """
        device = x.device
        b = x.shape[0]
        
        # 1. Random Timesteps Sampling
        # 0 ~ T-1 사이의 임의의 정수 추출
        t = torch.randint(0, self.schedule.timesteps, (b,), device=device).long()
        
        # 2. Loss Calculation
        # p_losses 내부에서 noise 주입(q_sample) 및 모델 예측 수행
        loss = p_losses(
            denoise_model=self.model,
            x_start=x,
            t=t,
            schedule=self.schedule,
            loss_type=self.loss_type
        )
        
        return loss
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Inference Step (Generation)
        shape: (Batch, Channel, Height, Width)
        """
        # p_sample_loop 내부에서 T부터 0까지 순차적으로 복원
        device = next(self.parameters()).device
        return p_sample_loop(
            model=self.model,
            shape=shape,
            schedule=self.schedule,
            device=device
        )
