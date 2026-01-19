import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

# Custom Modules
from dataset import get_dataloader
from model import Unet
from diffusion.ddpm import DDPM
from utils import setup_seed, save_images, EMA, prepare_logging

def train(args):
    # 1. Setup Device & Reproducibility
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device} with seed {args.seed}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")

    # 2. Logging Setup
    run_dir = os.path.join(args.result_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    sample_dir = os.path.join(run_dir, 'samples')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    logger = prepare_logging(args.run_name)
    
    # 3. Data Setup
    print("📚 Loading Dataset...")
    # dataset.py의 get_dataloader 사용 (다운로드 및 전처리 포함)
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Check Normalization (Safety)
    sample_img, _ = next(iter(dataloader))
    if sample_img.min() < -1.1 or sample_img.max() > 1.1:
        print(f"⚠️ Warning: Data range seems off. Min: {sample_img.min():.2f}, Max: {sample_img.max():.2f}")
        print("   - Expected range: [-1, 1]")
    else:
        print(f"✅ Data range verified: [{sample_img.min():.2f}, {sample_img.max():.2f}]")

    # 4. Model Setup
    print("🏗️ Initializing Model...")
    # Phase 1 Config (Base 64 channels)
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 2, 4),
        with_time_emb=True
    ).to(device)

    # DDPM Wrapper
    ddpm = DDPM(
        denoise_model=model,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        loss_type='l2'
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # EMA Setup (Optional)
    ema: Optional[EMA] = None
    if args.use_ema:
        ema = EMA(model, beta=0.9999)
        ema.register()
        print("   - EMA Enabled (beta=0.9999)")

    # Mixed Precision Setup
    scaler = GradScaler(enabled=args.amp)
    if args.amp:
        print("   - Mixed Precision (AMP) Enabled")

    # 5. Training Loop
    global_step = 0
    total_steps = args.max_steps
    
    print(f"🏁 Starting Training for {total_steps} steps...")
    
    while global_step < total_steps:
        for i, (images, _) in enumerate(dataloader):
            if global_step >= total_steps:
                break
            
            optimizer.zero_grad()
            
            images = images.to(device)
            
            # Forward & Loss
            with autocast(enabled=args.amp):
                # DDPM forward handles sampling t and noise injection
                loss = ddpm(images)
            
            # Backward & Optimization
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # EMA Update
            if ema is not None:
                ema.update()
            
            # Log
            if global_step % args.log_interval == 0:
                print(f"[Step {global_step}/{total_steps}] Loss: {loss.item():.4f}")
                logger.info(f"Step {global_step} Loss: {loss.item():.4f}")
            
            # Save Checkpoint
            if global_step % args.save_interval == 0 and global_step > 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step}.pt")
                save_dict = {
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args
                }
                if ema is not None:
                    save_dict['ema_state_dict'] = ema.shadow
                
                torch.save(save_dict, ckpt_path)
                print(f"💾 Checkpoint saved: {ckpt_path}")
                
            # Periodic Sampling
            if global_step % args.sample_interval == 0 and global_step > 0:
                print(f"✨ Sampling {args.num_samples} images at step {global_step}...")
                
                # EMA Model로 샘플링 (권장)
                eval_model = model
                if ema is not None:
                    ema.apply_shadow() # Apply EMA weights
                    
                # Sample
                with torch.no_grad():
                    # ddpm.sample 내부에서 p_sample_loop 호출 (tqdm 포함)
                    sampled_images = ddpm.sample(
                        shape=(args.num_samples, 3, 32, 32)
                    )
                
                # Save Image
                save_path = os.path.join(sample_dir, f"sample_step_{global_step}.png")
                # [-1, 1] -> [0, 1] 변환은 save_images 내부에서 처리하거나 여기서 처리
                # utils.save_images가 (B, C, H, W)를 받아 저장한다고 가정 (보통 0~1 or -1~1 예상)
                # 여기서는 명시적으로 [0, 1]로 변환하여 넘기는 것이 안전
                sampled_images = (sampled_images + 1) * 0.5
                sampled_images = torch.clamp(sampled_images, 0, 1)
                
                save_images(sampled_images, save_path)
                print(f"🖼️ Sample saved: {save_path}")
                
                # EMA 복원
                if ema is not None:
                    ema.restore()
            
            global_step += 1
            
    # Final Save
    final_ckpt_path = os.path.join(ckpt_dir, "ckpt_final.pt")
    save_dict = {
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    if ema is not None:
        save_dict['ema_state_dict'] = ema.shadow
    torch.save(save_dict, final_ckpt_path)
    print("🏆 Training Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM Training - CIFAR10")
    
    # Training Hyperparameters
    parser.add_argument('--run_name', type=str, default='ddpm_cifar10_exp1', help='Experiment name')
    parser.add_argument('--max_steps', type=int, default=800000, help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    
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
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--sample_interval', type=int, default=2000, help='Sample images every N steps')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of images to sample')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()
    
    # Boolean parsing correction
    args.use_ema = args.use_ema.lower() == 'true'
    
    train(args)
