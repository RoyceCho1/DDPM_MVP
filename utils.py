import os
import random
import torch
import numpy as np
import torchvision
from copy import deepcopy

def setup_seed(seed=42):
    """
    재현성을 위해 모든 랜덤 시드를 고정합니다.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, nrow=8, **kwargs):
    """
    [-1, 1] 범위의 텐서를 받아 [0, 1]로 변환 후 그리드 이미지로 저장합니다.
    """
    # 1. [-1, 1] -> [0, 1] Unnormalization
    # image data consists of integers... scaled linearly to [-1, 1] 
    grid = (images + 1) / 2
    grid = torch.clamp(grid, 0, 1) # 혹시 모를 오차 제거
    
    # 2. Make Grid & Save
    torchvision.utils.save_image(grid, path, nrow=nrow, **kwargs)

class EMA:
    """
    Exponential Moving Average for Model Parameters.
    DDPM 논문에서는 0.9999 decay를 사용했습니다.
    """
    def __init__(self, model, beta=0.9999):
        super().__init__()
        self.beta = beta
        self.step = 0
        # 모델의 복사본(Shadow) 생성
        self.ema_model = deepcopy(model).eval()
        
        # 파라미터 gradient 추적 끄기 (메모리 절약)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        현재 학습 모델의 파라미터를 EMA 모델에 반영합니다.
        """
        self.step += 1
        
        # state_dict를 순회하며 가중치 업데이트
        # new_average = beta * old_average + (1 - beta) * current_value
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.beta).add_(model_param.data, alpha=1 - self.beta)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

def prepare_logging(run_name="DDPM_Experiment"):
    """
    실험 결과를 저장할 디렉토리를 생성합니다.
    구조:
      ./results/run_name/
          ├── images/
          └── checkpoints/
    """
    os.makedirs("results", exist_ok=True)
    
    # 중복되지 않는 실험 폴더 이름 생성
    base_path = os.path.join("results", run_name)
    if not os.path.exists(base_path):
        run_path = base_path
    else:
        i = 1
        while True:
            run_path = f"{base_path}_{i}"
            if not os.path.exists(run_path):
                break
            i += 1
            
    img_path = os.path.join(run_path, "images")
    ckpt_path = os.path.join(run_path, "checkpoints")
    
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    
    print(f"✅ Experiment directories created at: {run_path}")
    return img_path, ckpt_path
