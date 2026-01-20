import torch
import torchvision
from dataset import get_dataloader
from utils import save_images

def inspect_data():
    print("🔍 Inspecting CIFAR-10 Dataset...")
    
    # 1. Load Batch
    dataloader = get_dataloader(batch_size=32, num_workers=2)
    images, labels = next(iter(dataloader))
    
    # 2. Print Stats
    print(f"✅ Batch Shape: {images.shape}") # Should be (32, 3, 32, 32)
    print(f"   - Min Value: {images.min().item():.4f}") # Should be close to -1
    print(f"   - Max Value: {images.max().item():.4f}") # Should be close to 1
    print(f"   - Mean Value: {images.mean().item():.4f}")
    print(f"   - Labels: {labels}")
    
    # 3. Save Preview
    # utils.save_images expects [-1, 1] input and converts to [0, 1] inside
    save_path = "dataset_preview.png"
    save_images(images, save_path, nrow=8)
    print(f"🖼️ Saved preview image to: {save_path}")
    print("   (Check this file to see real images)")

if __name__ == "__main__":
    inspect_data()
