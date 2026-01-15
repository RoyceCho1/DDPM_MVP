import torch
import torch.nn as nn
from diffusion.beta_schedule import make_ddpm_schedule
from diffusion.sampling import p_sample_loop

class DummyModel(nn.Module):
    """
    Sampling 테스트를 위한 더미 모델.
    노이즈를 예측하는 척 하지만, 실제로는 0에 가까운 값을 내뱉어서
    이미지가 점점 발산하지 않는지 확인용.
    """
    def forward(self, x, t):
        # 항상 작은 랜덤 값 리턴
        return torch.randn_like(x) * 0.1

def test_sampling():
    print("🧪 Testing Sampling Process...")
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    schedule = make_ddpm_schedule(timesteps=100, device=device) # 시간 절약을 위해 T=100
    model = DummyModel().to(device)
    
    # 2. Run Sampling Loop
    # (Batch=2, Channel=3, Height=16, Width=16[작게])
    shape = (2, 3, 16, 16)
    print(f"Generating image of shape {shape} with T={schedule.timesteps}...")
    
    generated_imgs = p_sample_loop(model, shape, schedule, device)
    
    # 3. Verify Output
    print(f"✅ Output Shape: {generated_imgs.shape}")
    
    if torch.isnan(generated_imgs).any():
        print("❌ Error: Output contains NaN!")
    elif torch.isinf(generated_imgs).any():
        print("❌ Error: Output contains Inf!")
    else:
        print(f"✅ Value Range: {generated_imgs.min().item():.3f} ~ {generated_imgs.max().item():.3f}")
        print("🎉 Sampling Test Passed!")

if __name__ == "__main__":
    test_sampling()
