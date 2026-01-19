import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    DDPM 논문에서 사용된 Sinusoidal Time Embedding.
    Timestep t를 고정된 sinusoidal 패턴의 벡터로 변환합니다.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # 3) Safety: sinusoidal input requires float
        time = time.float()
        device = time.device
        
        # 2) Safety: dim must be even for sin/cos split
        half_dim = self.dim // 2
        
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def safe_groups(ch, max_groups=32):
    """
    1) Safety: Find the largest divisor of 'ch' <= 'max_groups'
    This prevents Runtime Error when (ch % groups != 0)
    """
    g = min(max_groups, ch)
    while ch % g != 0:
        g //= 2
    return max(g, 1)

class Block(nn.Module):
    """
    기본적인 Conv Block: GroupNorm -> SiLU -> Conv (Pre-activation)
    """
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(safe_groups(dim, groups), dim)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x

class ResnetBlock(nn.Module):
    """
    DDPM U-Net의 핵심 블록.
    구조: Block1 -> Add Time Emb -> Block2 -> Residual Add
    """
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        
        # Time Embedding Projection
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
        else:
            self.mlp = None
            
        # Residual Connection
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            # Time Embedding을 feature map 크기에 맞춰 broadcasting 후 더함
            time_emb = self.mlp(time_emb)
            h = h + time_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    """
    해상도를 절반으로 줄임. (Stride Conv 사용)
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    해상도를 2배로 늘림. (Interpolate + Conv 사용, Checkerboard artifact 방지)
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        
    def forward(self, x):
        # Nearest Neighbor Interpolation -> Convolution
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

def default(val, d):
    """
    유틸리티 함수: val이 None이면 d를 반환
    """
    if val is not None:
        return val
    return d

class Unet(nn.Module):
    """
    Phase 1 U-Net (No Attention)
    - Base Channels: 64
    - Channel Mults: 1, 2, 2, 4
    """
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 2, 4),
        channels=3,
        with_time_emb=True
    ):
        super().__init__()
        self.channels = channels
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # e.g. [(64, 64), (64, 128), (128, 128), (128, 256)]

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Down Path
        # 각 레벨에서 (ResBlock -> ResBlock -> Downsample)
        # Skip connection 저장을 위해 각 레벨 출력 채널 기록 필요
        skip_dims = []
        cur_dim = init_dim
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
            
            skip_dims.append(dim_out)
            cur_dim = dim_out

        # Mid Block
        self.mid_block1 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)

        # Up Path
        skip_dims_reversed = list(reversed(skip_dims))
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            # Up stage 입력: (직전 Up 결과) + (Down에서의 Skip)
            # 여기의 dim_in은 사실 이전 단계의 dim_out (작은 채널)
            # dim_out은 더 큰 채널 (Down의 입력이었던 것)
            # 하지만 변수명이 reversed(in_out)이라 의미가 반대임.
            # dim_in: 128, dim_out: 256 (위 예시 기준)
            # 우리가 원하는건 256 -> 128.
            # 정확히는 (256 + 256) -> 128
            
            is_last = i >= (num_resolutions - 1)
            skip_dim = skip_dims_reversed[i] # dim_out 과 동일
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + skip_dim, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
            
        self.out_dim = default(out_dim, channels)
        self.final_res_block = ResnetBlock(init_dim, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(self, x, time):
        # 1. Init
        x = self.init_conv(x)
        r = x.clone()
        
        # 2. Time Emb
        t = self.time_mlp(time) if self.time_mlp is not None else None
        
        # 3. Down
        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x) # Skip connection 저장
            x = downsample(x)
            
        # 4. Mid
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # 5. Up
        for block1, block2, upsample in self.ups:
            # Skip connection pop
            skip = h.pop()
            x = torch.cat((x, skip), dim=1) # Channel Concatenate
            
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            
        # 6. Final
        x = self.final_res_block(x, t)
        return self.final_conv(x)
