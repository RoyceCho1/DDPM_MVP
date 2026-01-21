import math
import argparse
import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

# Custom Modules
from dataset import get_dataloader
from model import Unet
from diffusion.ddpm import DDPM
from utils import setup_seed, save_images, EMA, prepare_logging

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Linear Warmup + Cosine Annealing LR Scheduler
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train(args):
    # 1. 초기 설정 (Setup)
    setup_seed(args.seed) # 재현성을 위한 시드 고정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device} with seed {args.seed}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")

    # 2. 로깅 및 저장 경로 설정 (Logging)
    # results/실험이름/samples, results/실험이름/checkpoints 폴더 생성
    logger = prepare_logging(args.run_name)
    
    # We reconstruct the path based on assumption (or user should manually verify unique names)
    # Using the logger's file handler to find the actual directory is a robust way if available.
    if logger.handlers:
        run_file = logger.handlers[0].baseFilename
        run_dir = os.path.dirname(run_file)
    else:
        # Fallback if unconfigured (unlikely with prepare_logging)
        run_dir = os.path.join("results", args.run_name)
        
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "images")
    
    # ensure directories exist (redundant but safe)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # 3. 데이터 로드 (Data Loading)
    print("Loading Dataset...")
    # dataset.py의 get_dataloader 사용 (다운로드 및 전처리 포함)
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Safety Check: 데이터가 [-1, 1] 범위로 잘 정규화되었는지 확인
    # DDPM은 가우시안 노이즈(평균 0, 분산 1)를 다루므로 입력 데이터도 -1~1 범위여야 성능이 나옴
    sample_img, _ = next(iter(dataloader))
    if sample_img.min() < -1.1 or sample_img.max() > 1.1:
        print(f"Warning: Data range seems off. Min: {sample_img.min():.2f}, Max: {sample_img.max():.2f}")
    else:
        print(f"Data range verified: [{sample_img.min():.2f}, {sample_img.max():.2f}]")

    # 4. 모델 및 최적화 설정 (Model & Optimizer)
    print("Initializing Model...")
    model = Unet(
        dim=64,                # 기본 채널 수
        channels=3,            # RGB
        dim_mults=(1, 2, 2, 4),# 채널 증폭 비율 (64 -> 128 -> 128 -> 256)
        with_time_emb=True
    ).to(device)

    # DDPM Wrapper
    ddpm = DDPM(
        denoise_model=model,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        loss_type='l2'  #MSE Loss
    ).to(device)
    
    # Optimizer (Use ddpm.parameters for safety)
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr)
    
    # LR Scheduler (Warmup + Cosine)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        warmup_steps=args.warmup_steps, 
        total_steps=args.max_steps
    )

    # EMA (Exponential Moving Average)(Optional)
    # 학습 중인 모델 파라미터의 이동 평균을 별도로 저장.
    # 생성(Inference) 시에는 이 EMA 모델을 쓰는 것이 품질이 훨씬 좋음.
    ema: Optional[EMA] = None
    if args.use_ema:
        ema = EMA(model, beta=0.9999)
        ema.register()  #현재 모델의 파라미터 상태를 EMA에 등록
        print("   - EMA Enabled (beta=0.9999)")

    # Mixed Precision(AMP): 메모리 절약 및 속도 향상
    scaler = GradScaler(enabled=args.amp)
    if args.amp:
        print("   - Mixed Precision (AMP) Enabled")

    # 5. 학습 루프 (Training Loop)
    global_step = 0
    total_steps = args.max_steps
    
    print(f"Starting Training for {total_steps} steps...")
    
    ddpm.train() # Explicit Train Mode

    while global_step < total_steps:
        # Dataloader는 epoch 단위로 돌리지만, DDPM은 step 단위로 제어
        for i, (images, _) in enumerate(dataloader):
            if global_step >= total_steps:
                break
            
            # Efficient Zero Grad
            optimizer.zero_grad(set_to_none=True)
            
            images = images.to(device)
            
            # Forward & Loss
            with autocast(enabled=args.amp):
                loss = ddpm(images)
            
            # Loss Check (NaN/Inf)
            if not torch.isfinite(loss):
                print(f"Warning: Loss is {loss.item()} at step {global_step}. Skipping step.")
                scaler.update() 
                continue

            # Backward & Optimization
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                # Use ddpm.parameters()
                torch.nn.utils.clip_grad_norm_(ddpm.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Update LR
            
            # EMA Update
            if ema is not None:
                ema.update()
            
            # Log
            if global_step % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Step {global_step}/{total_steps}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                logger.info(f"Step {global_step} Loss: {loss.item():.4f} LR: {current_lr:.6f}")
            
            # Save Checkpoint
            if global_step % args.save_interval == 0 and global_step > 0:
                save_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                
                save_dict = {
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args
                }
                if ema is not None:
                    save_dict['ema_state_dict'] = ema.state_dict()
                
                torch.save(save_dict, save_path)
                print(f"Checkpoint saved: {save_path}")
                
            # Sampling
            if global_step % args.sample_interval == 0 and global_step > 0:
                print(f"Sampling {args.num_samples} images at step {global_step}...")
                
                # Eval Mode
                ddpm.eval()
                
                if ema is not None:
                    ema.apply_shadow()
                    
                with torch.no_grad():
                    sampled_images = ddpm.sample(
                        shape=(args.num_samples, 3, 32, 32)
                    )
                
                # Undo EMA
                if ema is not None:
                    ema.restore()
                
                ddpm.train() # Restore Train Mode

                # Save Image
                # [-1, 1] -> [0, 1]
                sampled_images = (sampled_images + 1) * 0.5
                save_path = os.path.join(sample_dir, f"sample_step_{global_step}.png")
                save_images(sampled_images, save_path) 
                print(f"Sample saved: {save_path}")
            
            global_step += 1
            
    # Final Save
    final_ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    save_dict = {
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    torch.save(save_dict, final_ckpt_path)
    print("Training Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM Training - CIFAR10")
    
    # Training Hyperparameters
    parser.add_argument('--run_name', type=str, default='ddpm_cifar10_exp1', help='Experiment name')
    parser.add_argument('--max_steps', type=int, default=800000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Warmup steps')
    
    # DDPM Hyperparameters
    parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)
    
    # Advanced Options
    parser.add_argument('--amp', action='store_true', help='Enable Mixed Precision Training')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--use_ema', type=str, default='true', choices=['true', 'false'], help='Use EMA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging & Saving
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--log_interval', type=int, default=100, help='Log loss every N steps')
    parser.add_argument('--save_interval', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--sample_interval', type=int, default=10000, help='Sample images every N steps')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to sample')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()
    
    args.use_ema = args.use_ema.lower() == 'true'
    
    train(args)
