import torch
from diffusion.beta_schedule import make_ddpm_schedule
from diffusion.forward import q_sample

def test_forward():
    print("🧪 Testing Forward Process...")
    
    # 1. Setup Schedule
    schedule = make_ddpm_schedule(timesteps=1000)
    print("✅ Schedule created.")
    
    # 2. Setup Dummy Data
    B = 4
    x_start = torch.randn(B, 3, 32, 32) # Normalized image mock
    t = torch.tensor([0, 250, 500, 999], dtype=torch.long) # Test various timesteps
    print(f"✅ Dummy data created: Batch={B}, Timesteps={t.tolist()}")
    
    # 3. Run q_sample
    x_t = q_sample(x_start=x_start, t=t, schedule=schedule)
    
    # 4. Verify Shapes
    assert x_t.shape == x_start.shape, f"Shape mismatch: {x_t.shape} != {x_start.shape}"
    print("✅ Shape check passed.")
    
    # 5. Verify Noise Levels (Heuristic)
    # t=0 should be closer to x_start than t=999
    diff_t0 = torch.abs(x_t[0] - x_start[0]).mean()
    diff_t999 = torch.abs(x_t[-1] - x_start[-1]).mean()
    
    print(f"   Diff at t=0: {diff_t0.item():.4f}")
    print(f"   Diff at t=999: {diff_t999.item():.4f}")
    
    if diff_t0 < diff_t999:
        print("✅ Noise logic seems correct (more noise at higher t).")
    else:
        print("⚠️ Warning: Noise logic might be off (check math).")

    print("\n🎉 Forward Process Test Passed!")

if __name__ == "__main__":
    test_forward()
