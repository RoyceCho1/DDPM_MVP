import torch
import torch.nn as nn
from diffusion.ddpm import DDPM

class DummyUNet(nn.Module):
    """
    테스트용 더미 U-Net
    입력과 동일한 크기의 출력을 내뱉음.
    Gradient Flow 확인을 위해 학습 가능한 파라미터를 하나 추가했습니다.
    """
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, t):
        # Time embedding etc.는 생략하고 단순히 출력 shape만 맞춤
        # 파라미터가 연산 그래프에 포함되어야 backward가 가능함
        return torch.randn_like(x) * self.dummy_param

def test_ddpm():
    print("🧪 Testing DDPM Wrapper...")
    
    # 1. Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # 더미 모델 생성
    unn = DummyUNet()
    
    # DDPM 초기화
    # 수정된 __init__ 시그니처 반영
    ddpm = DDPM(
        denoise_model=unn,
        timesteps=100, # 테스트용으로 작게 설정
    )
    ddpm.to(device) # .to(device) 호출 시 내부 버퍼들도 이동하는지 확인
    print("✅ DDPM Initialized and moved to device.")
    
    # 2. Test Training Step (Forward Path)
    B = 4
    x = torch.randn(B, 3, 32, 32).to(device)
    
    loss = ddpm(x)
    print(f"✅ Training Step Loss: {loss.item()}")
    
    # Gradient Check
    loss.backward()
    
    if unn.dummy_param.grad is not None:
        print("✅ Backward pass successful (Gradient computed).")
    else:
        print("❌ Backward pass failed (No Gradient).")
    
    # 3. Test Sampling Step (Reverse Path)
    print("Generating samples...")
    shape = (2, 3, 16, 16)
    
    # sample 함수 내부에서 device를 잘 추론하는지 확인
    samples = ddpm.sample(shape)
    
    print(f"✅ Generated Sample Shape: {samples.shape}")
    print(f"✅ Sample Device: {samples.device}")
    
    assert samples.device.type == device.type, f"Device mismatch! Expected {device}, got {samples.device}"
    
    print("🎉 DDPM Wrapper Test Passed!")

if __name__ == "__main__":
    test_ddpm()
