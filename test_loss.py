import torch
import torch.nn as nn
from diffusion.beta_schedule import make_ddpm_schedule
from diffusion.loss import p_losses

class DummyModel(nn.Module):
    """
    테스트를 위한 가짜 모델.
    Input과 동일한 Shape의 랜덤 텐서를 반환합니다.
    """
    def __init__(self):
        super().__init__()
        # 학습 가능한 파라미터 하나 추가 (Gradient check용)
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t):
        # x: (B, C, H, W)
        # t: (B,)
        # Output: (B, C, H, W) - Predicted Noise
        return torch.randn_like(x) * self.dummy_param

def test_loss():
    print("🧪 Testing Loss Function...")
    
    # 1. Setup
    schedule = make_ddpm_schedule(timesteps=1000)
    model = DummyModel()
    B = 4
    x_start = torch.randn(B, 3, 32, 32)
    t = torch.randint(0, 1000, (B,))
    
    # 2. Calculate Loss
    loss = p_losses(model, x_start, t, schedule, loss_type='l2')
    
    print(f"✅ Loss calculated: {loss.item()}")
    
    # 3. Check Backward (Gradient Flow)
    loss.backward()
    if model.dummy_param.grad is not None:
        print("✅ Gradient flow check passed.")
    else:
        print("❌ Gradient flow check failed!")

    # 4. Check L1 Loss
    loss_l1 = p_losses(model, x_start, t, schedule, loss_type='l1')
    print(f"✅ L1 Loss calculated: {loss_l1.item()}")
    
    print("\n🎉 Loss Function Test Passed!")

if __name__ == "__main__":
    test_loss()
