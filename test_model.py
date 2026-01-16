import torch
import torch.nn as nn
from model import Unet

def test_model():
    print("🧪 Testing U-Net Model (Phase 1)...")
    
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 초기화 (Base 64)
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 2, 4)
    ).to(device)
    
    print("✅ Model initialized.")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   - Parameter count: {param_count:,}")
    
    # 2. Input Setup
    B = 2
    x = torch.randn(B, 3, 32, 32).to(device)
    
    # Time input (Long type check)
    t = torch.randint(0, 1000, (B,), device=device).long()
    
    # 3. Forward Pass
    print(f"Forward pass with input {x.shape} and t {t.shape}...")
    try:
        out = model(x, t)
        print(f"✅ Output shape: {out.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 4. Check shape
    expected_shape = (B, 3, 32, 32)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    print("✅ Output shape correct.")
    
    # 5. Backward Pass (Gradient Flow Check)
    print("Checking gradient flow...")
    loss = out.mean()
    loss.backward()
    
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if len(gradients) > 0:
        print("✅ Backward pass successful (Gradients computed).")
    else:
        print("❌ Backward pass failed (No gradients found).")

    print("🎉 Model Test Passed!")

if __name__ == "__main__":
    test_model()
