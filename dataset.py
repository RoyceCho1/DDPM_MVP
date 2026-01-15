import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int = 128, num_workers: int = 2):
    """
    CIFAR-10 데이터셋을 다운로드하고, 전처리하여 DataLoader를 반환합니다.
    
    Transforms:
        - RandomHorizontalFlip: 데이터 증강
        - ToTensor: 이미지를 텐서로 변환 (0~1)
        - Normalize: (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) -> (-1 ~ 1) 범위로 스케일링
    """
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 학습용 데이터셋만 사용 (DDPM 학습)
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True # 배치 크기가 딱 떨어지지 않으면 나머지는 버림 (차원 오류 방지)
    )
    
    return dataloader

if __name__ == "__main__":
    # 테스트 코드
    dl = get_dataloader(batch_size=4)
    images, labels = next(iter(dl))
    print(f"Batch shape: {images.shape}")
    print(f"Value range: {images.min().item():.2f} ~ {images.max().item():.2f}")
