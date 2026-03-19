# DDPM (Denoising Diffusion Probabilistic Model) - CIFAR10

## 개요 (Overview)
이 프로젝트는 CIFAR-10 데이터셋을 활용하여 Denoising Diffusion Probabilistic Model (DDPM)을 구현하고 학습/샘플링하는 코드입니다. 추가적으로 더 빠른 샘플링 속도를 위해 DDIM (Denoising Diffusion Implicit Models) 샘플링 방식도 지원합니다.

## 프로젝트 구조 (Project Structure)
- `train.py`: DDPM 모델 학습 스크립트 (EMA(Exponential Moving Average), AMP(Automatic Mixed Precision) 지원)
- `sample.py`: 학습된 모델(Checkpoint)을 불러와 이미지를 생성하는 스크립트 (DDPM 및 DDIM 샘플링 지원)
- `model.py`: U-Net 아키텍처 구현체
- `diffusion/`: DDPM 및 DDIM 샘플링 관련 핵심 로직 (Forward process, Reverse process)
- `dataset.py`: CIFAR-10 데이터 로더 구성
- `utils.py`: 난수 시드(Seed) 고정, 이미지 저장 등의 유틸리티 모음
- `analyze_logs.py`: 학습 로그 분석 및 시각화용 스크립트

## 요구사항 (Requirements)
- PyTorch
- torchvision
- tqdm

---

## 사용 방법 (Usage)

### 1. 모델 학습 (Training)
`train.py`를 사용하여 모델 학습을 시작합니다. 로컬뿐만 아니라 원격 서버에서도 쉽게 실행할 수 있도록 다양한 파라미터를 지원합니다.

```bash
python train.py \
    --run_name ddpm_cifar10 \
    --max_steps 100000 \
    --batch_size 128 \
    --lr 2e-4 \
    --timesteps 1000 \
    --amp \
    --use_ema true
```

**[주요 파라미터]**
- `--run_name`: 실험 이름 (이 이름으로 로그와 체크포인트 폴더가 생성됨)
- `--max_steps`: 총 학습 스텝 수
- `--batch_size`: 한 번에 학습할 배치 사이즈
- `--timesteps`: Diffusion 과정에서의 T 스텝 수 (기본 1000)
- `--amp`: Mixed Precision Training 사용 여부 (VRAM 절약 및 속도 향상)
- `--use_ema`: EMA(Exponential Moving Average) 가중치 업데이트 사용 여부 (`true` 권장)
- `--save_interval`: 체크포인트 저장 주기

### 2. 이미지 생성 (Sampling)
학습을 통해 얻어진 체크포인트 파일(`.pt`)을 사용하여 `sample.py`로 새로운 이미지를 생성합니다.

#### 기본 DDPM 샘플링
정통 DDPM 방식으로 1,000 스텝을 거쳐 샘플링합니다.
```bash
python sample.py \
    --checkpoint ./logs/ddpm_cifar10/checkpoints/latest.pt \
    --output_dir ./generated \
    --num_samples 64 \
    --batch_size 32 \
    --method ddpm
```

#### 고속 샘플링 (DDIM)
DDIM 방식을 이용해 50스텝 만에 빠르게 이미지를 생성합니다.
```bash
python sample.py \
    --checkpoint ./logs/ddpm_cifar10/checkpoints/latest.pt \
    --num_samples 64 \
    --method ddim \
    --ddim_steps 50 \
    --eta 0.0
```

#### 노이즈 제거 과정 시각화 (Saving Intermediate Process)
완전한 노이즈 형태에서 점진적으로 이미지가 복원되는 전체 과정을 시각화하여 저장하고 싶을 때 사용합니다.
```bash
python sample.py \
    --checkpoint ./logs/ddpm_cifar10/checkpoints/latest.pt \
    --save_process \
    --process_interval 100
```

**[주요 파라미터]**
- `--checkpoint`: 학습이 완료된 체크포인트 파일의 경로 (필수)
- `--num_samples`: 생성할 전체 이미지 개수 (기본 64)
- `--method`: 샘플링 방식 선택 (`ddpm` 또는 `ddim`)
- `--ddim_steps`: DDIM을 사용할 때의 역추적 스텝 수 (기본 50)
- `--save_process`: 중간 Diffusion 과정을 이미지 격자로 저장할지 여부
- `--use_ema`: 생성 시 EMA 가중치를 적용할지 여부. 높은 하이퀄리티 샘플링을 위해 기본적으로 `true`가 권장됩니다.
