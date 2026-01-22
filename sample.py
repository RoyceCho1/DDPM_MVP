import argparse
import os
import torch
import math
from tqdm import tqdm

from model import Unet
from diffusion.ddpm import DDPM
from utils import setup_seed, save_images

def sample(args):
    # 1. 환경 설정 (Setup)
    # 실험 재현성을 위해 시드 고정
    setup_seed(args.seed)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Sampling on {device} with seed {args.seed}")
    
    # 2. 체크포인트 로드 (Load Checkpoint)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    print(f"Loading checkpoint: {args.checkpoint}")
    # map_location을 사용하여 저장된 기기와 다른 기기에서도 로드 가능하게 함
    ckpt = torch.load(args.checkpoint, map_location=device)
    # 학습 당시의 설정값들(args)을 가져옴
    train_args = ckpt['args']
    
    # 3. 모델 초기화 (Model Init)
    # train.py에서는 모델 구조를 하드코딩함:
    # dim=64, dim_mults=(1,2,2,4), channels=3, with_time_emb=True
    # 이 값들은 argparse args에 저장되지 않으므로,
    # inference 시 동일한 설정을 그대로 재현해야 함.
    dim = getattr(train_args, 'dim', 64)    # 안전장치용(실제로는 사용 안됨)
    
    # [Auto-Detection] Attention 사용 여부 확인
    # 체크포인트의 state_dict 키를 검사하여 'mid_attn'이나 'to_qkv'가 있는지 확인
    state_dict_to_check = ckpt['model_state_dict']
    has_attn = any('mid_attn' in k or 'to_qkv' in k for k in state_dict_to_check.keys())
    
    if has_attn:
        print("Detected Attention layers in checkpoint.")
    else:
        print("No Attention layers detected (Phase 1 checkpoint).")

    print("Initializing Model...")
    # train.py에서 하드코딩했던 값들을 그대로 사용
    model = Unet(
        dim=64,                  # train.py와 동일
        channels=3,              # CIFAR-10 RGB
        dim_mults=(1, 2, 2, 4),   # train.py와 동일
        with_time_emb=True,
        with_attn=has_attn     # 감지된 설정 적용
    ).to(device)

    # DDPM 설정 값 추출
    timesteps = getattr(train_args, 'timesteps', 1000)
    beta_start = getattr(train_args, 'beta_start', 1e-4)
    beta_end = getattr(train_args, 'beta_end', 0.02)
    
    # DDPM 초기화
    ddpm = DDPM(
        denoise_model=model,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        loss_type='l2'
    ).to(device)
    

    # 4. Load Weights (EMA vs Standard)
    loaded_ema = False
    
    # 사용자가 EMA 가중치 사용을 원하고, 체크포인트에 EMA 데이터가 있다면 우선 로드
    if args.use_ema:
        if 'ema_state_dict' in ckpt:
            print("Loading EMA weights for better quality...")
            # ddpm.model은 위에서 만든 model 객체를 가리키므로,
            # EMA 가중치를 이 객체에 로드함
            ddpm.model.load_state_dict(ckpt['ema_state_dict'])
            loaded_ema = True
        else:
            print("EMA weights requested but not found in checkpoint. Loading standard weights.")
            ddpm.model.load_state_dict(ckpt['model_state_dict'])
    else:
        # EMA를 끄거나 체크포인트에 없는 경우 일반 weight 로드
        print("Standard weights loaded (EMA disabled).")
        ddpm.model.load_state_dict(ckpt['model_state_dict'])
    
    # 모델을 평가 모드로 전환
    ddpm.eval()
    
    # 5. Generation Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_samples = args.num_samples
    # 전체 샘플 수를 배치 크기로 나눠서 몇 번 루프를 돌지 계산
    batches = math.ceil(total_samples / args.batch_size)
    all_images = []
    generated_count = 0
    
    print(f"Generating {total_samples} samples in {batches} batches...")
    
    with torch.no_grad():
        for i in tqdm(range(batches), desc="Sampling Batches"):
            # Calculate exact batch size (Bug Fix: use generated_count)
            current_bs = min(args.batch_size, total_samples - generated_count)
            
            # Sampling (reverse process) -> [-1, 1]
            capture_step = args.process_interval if args.save_process else None
            
            if args.method == 'ddpm':
                imgs = ddpm.sample(shape=(current_bs, 3, 32, 32), capture_every=capture_step)
            else: # ddim
                # ddim 방식(50 steps)
                imgs = ddpm.sample_ddim(
                    shape=(current_bs, 3, 32, 32),
                    ddim_steps=args.ddim_steps,
                    eta=args.eta,
                    capture_every=capture_step
                )
            
            all_images.append(imgs)
            generated_count += current_bs
            
    # Post-processing
    if args.save_process:
        # all_images: List of List[Tensor(B, C, H, W)] (Batches x TimeSteps)
        # We want to stack batches first.
        # List[Batch] -> List[TimeStep] -> Tensor(B...)
        # Let's flatten: For each batch, we have N steps.
        # Goal: Tensor (TotalSamples, TimeSteps, C, H, W) -> Grid
        
        # 1. Unpack batches
        # imgs_per_step: List[Tensor(TotalB, C, H, W)]
        num_steps = len(all_images[0])
        imgs_per_step = []
        for step_idx in range(num_steps):
             # Collect step_idx-th tensor from all batches
             step_batches = [batch_list[step_idx].to('cpu') for batch_list in all_images]
             step_batch = torch.cat(step_batches, dim=0)
             imgs_per_step.append(step_batch)
             
        # 2. Make Grid
        # We want rows = Random Samples, Cols = Time Steps
        # Each row: T_0, T_100, ..., T_1000 (Data)
        # imgs_per_step[0] is T=1000 (Noise)? No, loop is T-1 to 0.
        # p_sample_loop logic:
        # capture every 100 on loop i (T-1..0)
        # If capture_every=100: i=900, 800.., 0.
        # Order in list: High Noise -> Low Noise (Data at end).
        
        # Grid construction:
        # For each sample in batch:
        #   [Img_step0, Img_step1, ... Img_final]
        
        final_grid_list = []
        for sample_idx in range(args.num_samples):
            row_imgs = [step_imgs[sample_idx] for step_imgs in imgs_per_step]
            final_grid_list.extend(row_imgs)
            
        all_images = torch.stack(final_grid_list)
        args.grid_nrow = num_steps # Force row width to match steps
        
    else:
        # List of Tensors(B, C, H, W) -> Tensor(TotalB, C, H, W)
        all_images = [img.to('cpu') for img in all_images]
        all_images = torch.cat(all_images, dim=0)

    # [-1, 1] -> [0, 1] Conversion (Consistency)
    all_images = (all_images + 1) * 0.5
    
    # 6. Save Result
    save_filename = f"sample_{args.num_samples}_seed{args.seed}"
    if loaded_ema:
        save_filename += "_ema"
    
    if args.save_process:
        save_filename += "_process"
        
    save_filename += ".png"
    
    save_path = os.path.join(args.output_dir, save_filename)
    
    print(f"Saving grid image to {save_path}...")
    # save_images now expects [0, 1]
    save_images(all_images, save_path, nrow=args.grid_nrow)
    
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM Inference - CIFAR10")
    
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (.pt)')
    parser.add_argument('--output_dir', type=str, default='./generated', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=64, help='Total number of images to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--grid_nrow', type=int, default=8, help='Images per row in grid')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_ema', type=str, default='true', choices=['true', 'false'], help='Use EMA weights if available')
    parser.add_argument('--save_process', action='store_true', help='Save intermediate diffusion steps')
    parser.add_argument('--process_interval', type=int, default=100, help='Interval for saving intermediate steps')
    
    # DDIM Arguments
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'ddim'], help='Sampling method')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of steps for DDIM')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM stochasticity (0.0=deterministic, 1.0=DDPM)')
    
    args = parser.parse_args()
    args.use_ema = args.use_ema.lower() == 'true'
    
    sample(args)
