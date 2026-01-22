from __future__ import annotations
import torch
from tqdm import tqdm
from diffusion.utils import extract

@torch.no_grad()
def p_sample(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    t: torch.Tensor, 
    t_index: int, 
    schedule
) -> torch.Tensor:
    """
    Reverse Diffusion Step: x_t -> x_{t-1}
    논문의 Algorithm 2, Step 4에 해당합니다.
    
    수식 (Eq. 11):
    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta) + sigma_t * z

    Args:
        model: 노이즈 예측 모델(epsilon_theta)
        x: 현재 이미지 텐서 (x_t)
        t: 현재 timestep (t)
        t_index: 현재 timestep 인덱스
        schedule: DiffusionSchedule 객체
    """

    # 1. 필요 계수 추출(Coefficient Extraction)
    betas_t = extract(schedule.betas, t, x.shape) #beta_t
    sqrt_one_minus_alphas_cumprod_t = extract(schedule.sqrt_one_minus_alphas_cumprod, t, x.shape) #sqrt(1-alpha_bar_t)
    sqrt_recip_alphas_t = extract(schedule.sqrt_recip_alphas, t, x.shape) #sqrt(1/alpha_t)
    
    # 2. 모델 예측 (Predicted Noise)
    # epsilon_theta(x_t, t) : 모델이 예측한 noise
    pred_noise = model(x, t)

    # 3. 평균 계산
    # mu_theta(x_t, t) : 모델이 예측한 평균
    # 수식 : mu = 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_bar)*epsilon_theta)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
    )

    # 4. 최종 sampling(mean + variance)
    # t=0 일 때는 noise 추가하지 않음
    if t_index == 0:
        return model_mean
    else:
        # variance 선택 : sigma_t^2 = beta_t (or beta_tilde_t)
        # beta_tilde_t로 선택
        # 수치 안정성을 위해 log variance를 사용 : exp(0.5 * log_var) == sqrt(var)
        posterior_log_variance_t = extract(schedule.posterior_log_variance_clipped, t, x.shape)

        # z ~ N(0, I)(Random Noise)
        noise = torch.randn_like(x)
        
        # x_{t-1} = mu + sigma * z
        return model_mean + torch.exp(0.5 * posterior_log_variance_t) * noise

@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module, 
    shape: tuple, 
    schedule, 
    device: torch.device,
    capture_every: int = None
) -> torch.Tensor | list[torch.Tensor]:
    """
    Full Reverse Process (Algorithm 2)
    x_T ~ N(0, I) 부터 시작해서 x_0까지 순차적으로 복원.
    capture_every (int): 지정된 스텝마다 이미지를 저장하여 리스트로 반환 (Visualizing Process용)
    """
    b = shape[0]
    
    # [Visualization] 중간 과정 저장용 리스트
    imgs = []

    # Step 1: 완전 Gaussian Noise에서 시작
    img = torch.randn(shape, device=device)
    
    # Step 2: Iterate from T-1 down to 0
    # tqdm으로 진행 상황 표시
    for i in tqdm(reversed(range(0, int(schedule.timesteps))), desc='Sampling loop time step', total=int(schedule.timesteps)):
       
        # 현재 배치의 타임스텝 t 텐서 생성 (batch size만큼 같은 값)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        # Step 3,4(Algorithm 2): Sample x_{t-1} from x_t
        img = p_sample(
            model=model, 
            x=img, 
            t=t, 
            t_index=i, 
            schedule=schedule
        )

        # [Visualization] Capture intermediate step
        if capture_every is not None and i % capture_every == 0:
            imgs.append(img)
            
    # [Visualization] Return logic
    if capture_every is not None:
        imgs.append(img) # Add final x_0
        return imgs # List of Tensors

    # Step 5: Return x_0   
    return img
