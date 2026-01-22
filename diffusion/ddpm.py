from __future__ import annotations
import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple

# 1. Type Hinting을 위한 클래스 import
# (IDE에서 schedule 객체의 속성을 자동완성 하거나 타입 체크를 할 수 있게 도와줍니다)
from diffusion.beta_schedule import make_ddpm_schedule, DiffusionSchedule
from diffusion.loss import p_losses
from diffusion.sampling import p_sample_loop, ddim_sample_loop

class DDPM(nn.Module):
    """
    DDPM (Denoising Diffusion Probabilistic Models) Wrapper Class
    이 클래스는 DDPM의 학습과 추론(샘플링) 전체 과정을 관장하는 컨트롤 타워입니다.
    
    역할
    1. Model Management: 노이즈를 예측하는 U-Net 모델(denoise_model)을 관리합니다.
    2. Schedule Management: 확산 과정에 필요한 상수들(beta, alpha 등)을 생성하고 관리합니다.
       - register_buffer를 사용해 GPU 이동(cuda())과 저장(state_dict)을 자동화합니다.
    3. Training Interface (forward): 학습 데이터를 받아 Loss를 계산합니다.
    4. Inference Interface (sample): 노이즈에서 시작해 이미지를 생성합니다.
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
        
        # 1. 모델 등록
        # 내부적으로 'model'이라는 이름을 사용하여 loss.py 등의 함수와 호환성을 맞춥니다.
        self.model = denoise_model
        self.loss_type = loss_type
        
        # 2. Diffusion Schedule 생성
        # 초기에는 CPU에서 계산합니다. (make_ddpm_schedule은 모든 계수가 담긴 객체를 반환)
        _schedule = make_ddpm_schedule(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=torch.device('cpu')
        )
        
        # 3. Schedule 데이터의 Buffer 등록 (핵심 로직)
        # 일반 변수(self.betas)가 아닌 PyTorch의 'Buffer'로 등록해야 하는 이유:
        #   a. model.cuda() 호출 시 스케줄 값들도 자동으로 GPU로 따라갑니다.
        #   b. model.state_dict() 저장 시 스케줄 값들도 함께 저장되어, 나중에 정확한 복원이 가능합니다.
        
        # 3-1. Timesteps (int -> Tensor 변환)
        # 스케줄 객체의 timesteps는 int지만, 버퍼 등록을 위해 Tensor로 변환합니다.
        # 이름 충돌 방지를 위해 내부적으로 사용하는 버퍼는 등록하되, 
        # 나중에 schedule 프로퍼티에서 다시 int로 복원해줄 것입니다.
        self.register_buffer('timesteps', torch.tensor(_schedule.timesteps))

        # 3-2. 나머지 텐서들 (betas, alphas, ...)
        # _schedule 객체 내부의 모든 텐서를 순회하며 버퍼로 등록합니다.
        for key, value in _schedule.__dict__.items():
            if torch.is_tensor(value):
                self.register_buffer(key, value)

        # 4. 구조 복원용 템플릿 저장
        # _schedule에는 CPU 텐서들이 들어있지만, 나중에 'schedule' 프로퍼티를 통해
        # 현재 디바이스에 맞는 텐서들로 교체된 새로운 객체를 만들 때 틀(Template)로 사용합니다.
        self._schedule_struct = _schedule

    @property
    def schedule(self) -> DiffusionSchedule:    
        """
        [Dynamic Schedule Reconstruction]
        현재 모델이 위치한 디바이스(CPU/GPU)에 있는 버퍼 값들을 모아서,
        기능 함수(p_losses, p_sample)들이 사용할 수 있는 'DiffusionSchedule' 객체로 즉석에서 조립하여 반환합니다.
        
        Returns:
            DiffusionSchedule: 현재 디바이스의 텐서들을 포함한 스케줄 객체
        """
        # 1. 템플릿 복사 (중요: copy를 사용하여 원본 _schedule_struct 훼손 방지)
        # 이전 코드에서는 원본을 수정하는 부작용(Side Effect)이 있었으나 이를 해결함.
        current_schedule = copy.copy(self._schedule_struct)
        
        # 2. 버퍼 값 덮어쓰기
        # self에 등록된 버퍼들(self.betas 등)은 현재 모델이 있는 디바이스(예: GPU)에 있습니다.
        # 이 값들을 가져와서 current_schedule의 속성으로 설정합니다.
        for key in current_schedule.__dict__.keys():
            if hasattr(self, key):
                val = getattr(self, key)
                # [Fix] timesteps buffer는 Tensor이므로, 다시 int로 변환해줍니다.
                # 이렇게 하면 sampling.py 등에서 range()를 쓸 때 에러가 나지 않습니다.
                if key == 'timesteps' and torch.is_tensor(val):
                    val = int(val.item())
                
                setattr(current_schedule, key, val)
                
        return current_schedule
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [Training Step - Algorithm 1]
        학습 루프(train.py)에서 호출되는 함수입니다.
        
        Process:
        1. 랜덤한 Timestep t를 샘플링합니다.
        2. 원본 이미지 x에 대해 Loss를 계산합니다.
        
        Args:
            x: (Batch, Channel, Height, Width) - 원본 이미지 (Normalized to [-1, 1])
        Returns:
            loss(Scalar): 학습 손실 (MSE)
        """
        # 입력 데이터가 있는 디바이스를 따릅니다.
        device = x.device
        b = x.shape[0]
        
        # 1. Random Timesteps Sampling
        # 0 ~ T-1 사이에서 배치 사이즈만큼 랜덤한 정수 t를 뽑습니다.
        # 예: [5, 12, 34, ...]
        t = torch.randint(0, self.schedule.timesteps, (b,), device=device).long()
        
        # 2. Loss Calculation (diffusion/loss.py)
        # - q_sample: 원본 x에 노이즈를 섞어 x_t를 만듭니다.
        # - model(x_t, t): 모델이 x_t를 보고 추가된 노이즈를 예측합니다.
        # - loss: 실제 노이즈와 예측된 노이즈 간의 차이(MSE)를 계산합니다.
        loss = p_losses(
            denoise_model=self.model,   # loss.py의 인자 이름 호환
            x_start=x,
            t=t,
            schedule=self.schedule,     # 현재 디바이스에 맞는 스케줄 객체 전달
            loss_type=self.loss_type
        )
        
        return loss
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], capture_every: int = None) -> torch.Tensor | list[torch.Tensor]:
        """
        [Inference Step - Algorithm 2]
        새로운 이미지를 생성할 때 호출되는 함수입니다.
        학습된 모델을 사용하여 완전한 노이즈로부터 이미지를 복원합니다.
        
        Args:
            shape: 생성할 이미지의 형태 (Batch, Channel, Height, Width)
            capture_every: 중간 과정 캡처 간격 (int or None)
        Returns:
            Generated Images (Tensor): [-1, 1] 범위로 생성된 이미지 (or List of Images)
        """
        # 모델 파라미터가 위치한 디바이스를 확인합니다. (CPU or CUDA)
        device = next(self.parameters()).device

        # 3. Full Reverse Process (diffusion/sampling.py)
        # 랜덤 노이즈 x_T부터 시작해, t = T-1, ..., 0 까지 순차적으로 노이즈를 제거합니다.
        # 결과적으로 x_0 (생성된 이미지)를 얻게 됩니다.
        return p_sample_loop(
            model=self.model,
            shape=shape,
            schedule=self.schedule,
            device=device,
            capture_every=capture_every
        )

    @torch.no_grad()
    def sample_ddim(
        self, 
        shape: Tuple[int, ...], 
        ddim_steps: int = 50, 
        eta: float = 0.0, 
        capture_every: int = None
    ) -> torch.Tensor | list[torch.Tensor]:
        # DDIM Sampling을 수행합니다.
        device = next(self.parameters()).device
        
        return ddim_sample_loop(
            model=self.model,
            shape=shape,       
            schedule=self.schedule,
            device=device,
            ddim_steps=ddim_steps,
            eta=eta,
            capture_every=capture_every
        )

