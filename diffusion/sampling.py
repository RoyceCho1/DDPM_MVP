from __future__ import annotations
import torch
from tqdm import tqdm
from diffusion.utils import extract
import numpy as np

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

def compute_alpha_bars(schedule, t):
    """
    임의의 시점 t에 대한 누적 alpha_bar 추출
    DDIM은 t가 999, 950, 900, ... 처럼 건너뛰므로
    해당 시점의 정확한 alpha_cumprod를 추출해야 한다
    """
    return extract(schedule.alphas_cumprod, t, t.shape)

@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    schedule,
    eta: float = 0.0
) -> torch.Tensor:
    """
    x_t, epsilon_theta를 이용해 x_{t_prev}를 복원
    """

    alpha_bar_t = compute_alpha_bars(schedule, t)
    alpha_bar_t_prev = compute_alpha_bars(schedule, t_prev)
    denom = torch.sqrt(1 - alpha_bar_t)
    
    # 모델이 예측한 noise
    epsilon_theta = model(x, t)
    
    # x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    # DDPM과 달리 DDIM은 매 스텝마다 'x_0'를 예측한다.
    pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / denom
    
    # 예측된 x_0가 이미지 범위(-1, 1)를 벗어나는 경우 clipping
    pred_x0 = torch.clamp(pred_x0, -1., 1.)
    
    # 분산 계산
    # sigma_t = eta * sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
    # eta = 0 이면 DDIM(같은 seed면 항상 같은 이미지 생성), eta = 1 이면 DDPM
    sigma_t = eta * torch.sqrt(
        (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
    )
    
    # x_t 방향 성분 계산(direction)
    # direction = sqrt(1 - alpha_bar_t_prev - sigma_t^2) * epsilon_theta
    pred_dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * epsilon_theta
    
    # Random Noise
    # eta = 0 이면 noise항이 사라져서 deterministic sampling
    if eta == 0:
        noise = 0
    else:
        noise = torch.randn_like(x)
        
    # x_{t-1} = sqrt(alpha_bar_t_prev) * pred_x0 + direction + noise
    # 수식 : "x_0 성분" + "방향 성분" + "random noise 성분"
    x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt + sigma_t * noise
    
    return x_prev

@torch.no_grad()
def ddim_sample_loop(
    model: torch.nn.Module,
    shape: tuple,
    schedule,
    device: torch.device,
    ddim_steps: int = 50,
    eta: float = 0.0,
    capture_every: int = None
) -> torch.Tensor | list[torch.Tensor]:
    """
    DDIM Sampling Loop
    전체 timesteps(1000)를 다 계산하는 것이 아니라, ddim_steps만큼의 stride(건너뛰면서)로 계산
    """
    b = shape[0]
    imgs = []
    
    # random noise에서 시작
    img = torch.randn(shape, device=device)
    
    # timestep 선정
    # 예: T=1000, ddim_steps=50 -> [0, 20, 40, ..., 980](총 50 step)
    total_steps = schedule.timesteps
    
    # 0 부터 T-1 까지 ddim_steps 개수만큼 숫자를 뽑는다
    times = np.linspace(0, total_steps - 1, ddim_steps).astype(int)

    # reverse process이므로 역순으로 정렬
    times = list(reversed(times))
    
    # (current timestep, next timestep) pair 생성
    # ex: [980, 960, ..., 20, 0] -> [(980, 960), (960, 940), ..., (20, 0), (0, -1)]
    # 마지막 pair는 (0, -1)은 0번 timestep에서 clean image로 복원하는 과정
    time_pairs = list(zip(times[:-1], times[1:])) 
    time_pairs.append((times[-1], -1))
    
    for i, (t_curr, t_next) in enumerate(tqdm(time_pairs, desc='DDIM Sampling')):
        # 현재 시점 t에 대한 tensor 생성
        t = torch.full((b,), t_curr, device=device, dtype=torch.long)
        
        if t_next < 0:
            pass
        
        # t_prev tensor 생성
        t_prev = torch.full((b,), max(0, t_next), device=device, dtype=torch.long)
        # 현재 시점의 누적 alpha 값 추출(alpha_bar_t)
        alpha_bar_t = extract(schedule.alphas_cumprod, t, shape)
        
        # 다음 시점의 누적 alpha 값 추출 및 예외 처리
        if t_next < 0:
            # t_next가 -1인 경우, alpha_bar_t_prev = 1(노이즈 없는 원본 이미지 생성)
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
        else:
            # t_next가 -1이 아닌 경우, 다음 시점의 누적 alpha 값 추출
            alpha_bar_t_prev = extract(schedule.alphas_cumprod, t_prev, shape)
             
        # 2. Predict Noise
        epsilon_theta = model(img, t)
        
        # 3. Predict x_0
        # 수식: x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_bar_t)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)
        
        # 4. Compute Variance
        # eta에 따라 결정된다.
        # eta = 0 이면 DDIM(같은 seed면 항상 같은 이미지 생성) eta = 1 이면 DDPM
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
        )
        
        # 5. Compute Direction
        pred_dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * epsilon_theta
        
        # 6. Noise
        if eta == 0:
            noise = 0
        else:
            noise = torch.randn_like(img)
            
        # 7. Update
        # 수식: x_{t-1} = (원본 성분) + (방향 성분) + (random noise 성분)
        img = torch.sqrt(alpha_bar_t_prev) * pred_x0 + pred_dir_xt + sigma_t * noise
        
        if capture_every is not None and i % capture_every == 0:
            imgs.append(img)
            
    if capture_every is not None:
        imgs.append(img)
        return imgs
        
    return img

