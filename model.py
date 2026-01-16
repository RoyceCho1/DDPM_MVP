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
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """
    기본적인 Conv Block: Conv -> GroupNorm -> SiLU
    """
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        # Safe GroupNorm: 채널 수가 32보다 작을 경우를 대비
        self.norm = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
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
        in_out = list(zip(dims[:-1], dims[1:])) # [(64, 64), (64, 128), (128, 128), (128, 256)]

        # Time Embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Down Sampling
        now_dim = init_dim
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # Skip connection 때문에 채널 수가 늘어남을 대비할 중간 변수
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Up Sampling
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i >= (num_resolutions - 1)

            # Skip connection concate으로 인해 입력 채널은 dim_out * 2 (혹은 설계따라 다름)
            # 여기서는 (dim_out + dim_in) -> dim_in 으로 가는 구조.
            # Upsample의 dims 리스트는 Down의 역순. 
            # Down: 64->64, 64->128, 128->128, 128->256
            # Up:   256->128, 128->128, 128->64, 64->64
            
            # 정확한 채널 매칭을 위해 Down block의 출력 채널을 잘 봐야 함.
            # Decoder의 입력은 (현재 레벨의 Up 채널) + (Encoder에서 넘어온 Skip 채널)
            # Encoder [64, 128, 128, 256] (각 단계 출력)
            
            # 여기서 dim_in은 원래 Down의 입력, dim_out은 출력.
            # Up에서는 dim_out(Down의 출력)이 입력이 되고 dim_in(Down의 입력)으로 복원해야 함.
            # reversed in_out: (128, 256), (128, 128), (64, 128), (64, 64) ?? 아니지.
            # in_out        : (64, 64), (64, 128), (128, 128), (128, 256)
            # reversed      : (128, 256), (128, 128), (64, 128), (64, 64)
            # Up Loop       : i=0. in=128, out=256 ??
            # 구조적으로 [Input] -> [Enc1] -> [Enc2] -> [Mid] -> [Dec2(+Enc2)] -> [Dec1(+Enc1)] -> Output
            
            # 통상적인 U-Net:
            # Enc: C -> C1 -> C2 -> C3
            # Mid: C3 -> C3
            # Dec: C3 + C3(Skip) -> C2 ...
            
            # 간단하게 다시 짭니다.
            pass

        # 편의를 위해 다시 작성.
        
def default(val, d):
    if val is not None:
        return val
    return d

class Unet(nn.Module):
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
                nn.GELU(),
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
                # Downsample을 하면 채널은 유지하고 H,W만 줄임 (여기 구현상)
                # 만약 Downsample에서 채널을 늘리는 구조라면 수정 필요.
                # 위 Downsample 클래스는 dim -> dim 임.
            ]))
            
            # Skip connection은 ResBlock을 통과한 후의 feature map들.
            # 모델 forward시 2개의 아웃풋이 나옴 (block1후? block2후?)
            # 보통 각 Down stage의 최종 출력을 skip으로 씀.
            # 여기선 block2 후, downsample 전의 값을 skip으로 쓴다고 가정.
            # 그러면 skip 채널은 dim_out.
            # 마지막 layer(is_last)는 downsample이 identity이므로 그대로.
            
            skip_dims.append(dim_out)
            cur_dim = dim_out

        # Mid Block
        self.mid_block1 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)

        # Up Path
        # Down의 역순.
        # in_out reversed: [(128, 256), (128, 128), (64, 128), (64, 64)]
        # 근데 Upsample은 (현재 차원 + 스킵 차원) -> (목표 차원) 이어야 함.
        
        skip_dims_reversed = list(reversed(skip_dims))
        
        # in_out을 그냥 쓰기보다는 dims를 보고 판단하는게 나음.
        # dims: [64, 64, 128, 128, 256]
        # Up Process: 256 -> 128 -> 128 -> 64 -> 64 ...
        
        # Up stage 입력: (직전 Up 결과) + (Down에서의 Skip)
        # 직전 Up 결과 채널: dim_out (이전 루프의)
        # Skip 채널: skip_dims_reversed[i]
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            # dim_in: 128, dim_out: 256 (from reversed in_out) -> 이건 아님.
            # Down이 A->B 였다면 Up은 (B+B) -> A 가 되어야 함 (Concatenation)
            # B (From main path) + B (From skip path) = 2B -> A
            
            # 정확히 역순으로 가야함.
            # Down Levels:
            # 0: 64 -> 64
            # 1: 64 -> 128
            # 2: 128 -> 128
            # 3: 128 -> 256
            
            # Up Levels:
            # 0: 256 + 256(Skip 3) -> 128 (매칭되는 Down 3의 입력) 
            # 1: 128 + 128(Skip 2) -> 128
            # 2: 128 + 128(Skip 1) -> 64
            # 3: 64 + 64(Skip 0) -> 64
            
            is_last = i >= (num_resolutions - 1)
            
            # Down Stage i: In(dim_in) -> Out(dim_out)
            # Up Stage i: In(dim_out) + Skip(dim_out) -> Out(dim_in)
            
            skip_dim = skip_dims_reversed[i] # dim_out과 같음
            
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
