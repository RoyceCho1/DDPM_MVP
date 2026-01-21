import os
import logging
import random
import torch
import numpy as np
import torchvision
from copy import deepcopy

def setup_seed(seed=42):
    """
    [Reproducibility Setting]
    실험의 재현성을 위해 사용되는 모든 랜덤 시드를 고정합니다.
    같은 시드에서는 항상 같은 랜덤 결과가 나와야 디버깅이 쉽습니다.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, nrow=8, **kwargs):
    """
    [Image Saving Helper]
    모델이 생성한 [-1, 1] 범위의 텐서를 사람이 볼 수 있는 [0, 1] 이미지 파일로 저장합니다.
    """
    # 1. Clamping
    # 수치 오차로 인해 0보다 작거나 1보다 큰 값이 생길 수 있으므로 잘라낸다 (Input should be [0, 1])
    grid = torch.clamp(images, 0, 1)
    
    # 3. Make Grid & Save
    # 배치 이미지들을 보기 좋게 Grid로 묶어서 저장
    torchvision.utils.save_image(grid, path, nrow=nrow, **kwargs)

class EMA:
    """
    [Exponential Moving Average]
    학습된 모델 파라미터의 이동 평균을 관리하는 클래스입니다.
    
    * 역할:
      1. 학습 중 파라미터의 급격한 변동을 완화.
      2. Inference(샘플링) 시 더 나은 품질의 이미지를 생성.
      3. weight swapping (Shadow <-> Original) 기능 지원.
    """
    def __init__(self, model, beta=0.9999):
        super().__init__()
        self.model = model
        self.beta = beta
        self.shadow = {}    # EMA 가중치를 저장할 dictionary
        self.backup = {}    # 원본 가중치를 백업할 dictionary
        
    def register(self):
        """
        [Initialization]
        학습 시작 전, 현재 모델의 파라미터를 복사하여 EMA 초기값을 설정합니다.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """
        [Step Update]
        매 학습 스텝마다 호출되어 이동 평균을 업데이트.
        Formula: V_t = beta * V_{t-1} + (1 - beta) * theta_t
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # beta 비율만큼은 과거 값을 유지, 1-beta는 현재 값을 반영
                new_average = (1.0 - self.beta) * param.data + self.beta * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        """
        [Swap to EMA: For Inference]
        샘플링(이미지 생성)을 하기 위해, 
        1) 현재 학습 중인 모델 파라미터를 backup에 저장하고
        2) 모델의 파라미터를 EMA 가중치로 교체
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        """
        [Restore Original: For Training]
        샘플링이 끝난 후, 다시 학습을 진행하기 위해
        backup해두었던 원래 파라미터로 모델을 복구합니다.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}    # 백업 비우기

    def state_dict(self):
        # 체크포인트 저장
        return self.shadow
        
    def load_state_dict(self, state_dict):
        # 체크포인트 로드
        self.shadow = state_dict

def prepare_logging(run_name="DDPM_Experiment"):
    """
    [Directory Setup]
    실험 결과를 체계적으로 저장하기 위해 폴더 구조를 생성합니다.
    이미 존재하면 _1, _2 등을 붙여 덮어쓰기를 방지합니다.
    
    Structure:
      ./results/run_name/
          ├── images/       (생성된 샘플 이미지)
          └── checkpoints/  (모델 가중치 .pt 파일)
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

    # 간단한 파일 로거 설정
    logging.basicConfig(
        filename=os.path.join(run_path, "train.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print(f"Experiment directories created at: {run_path}")
    return logger
