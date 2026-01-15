import torch
from diffusion.beta_schedule import extract

# 1. 더미 데이터 준비
a = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])  # 전체 시간표 (T=5)
t = torch.tensor([1, 4])                     # 배치 내 이미지들의 시점
x_shape = (2, 3, 4, 4)                       # 이미지 모양 (작게 설정)

# 2. extract 실행
out = extract(a, t, x_shape)

print(f"입력 t: {t}")
print(f"추출된 값: {a[t]}")  # [0.1, 0.4]
print(f"변환된 모양: {out.shape}")
print(f"실제 데이터:\n{out}")