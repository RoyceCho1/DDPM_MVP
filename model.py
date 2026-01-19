import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Helper Modules (기본 부품들)
# ==============================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    [Time Embedding Generator]
    DDPM은 noise 강도(t)에 따라 동작이 달라진다.
    정수형 시점 t(0~999)를 transformer처리 고차원 벡터(sin/cos)로 변환
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Safety: sinusoidal 연산을 위해 float형으로 변환
        time = time.float()
        device = time.device
        
        # Safety: sin/cos을 반반씩 나누기 위해 짝수여야 함
        half_dim = self.dim // 2
        
        # 10000^*(2i/d_model) 수식 구현(log space에서 계산하여 안정성 확보)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # (Batch, 1) * (1, Dim/2) -> (Batch, Dim/2) Broadcasting
        embeddings = time[:, None] * embeddings[None, :]

        # sin, cos을 합쳐서 최종 임베딩 생성 -> (Batch, Dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def safe_groups(ch, max_groups=32):
    """
    [Utility] GroupNorm 에러 방지용
    GroupNorm은 채널 수가 그룹 수로 나누어 떨어져야 한다
    채널 수(Ch)가 32보다 작거나 나누어 떨어지지 않을 때, 가장 적절한(약수) 그룹 수를 찾는다.
    """
    g = min(max_groups, ch)
    while ch % g != 0:
        g //= 2
    return max(g, 1)

class Block(nn.Module):
    """
    [Basic Convolutional Block]
    구조: GroupNorm -> SiLU -> Conv2d
    * 특징: Pre-activation 구조 (Norm/Act가 Conv 앞에 옴)를 사용하여 학습 안정성을 높임.
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
    [Core Component: Time-Conditioned ResNet Block]
    DDPM U-Net의 핵심 블록입니다.
    이미지 특징(x)에 시간 정보(time_emb)를 주입(Injection)하여,
    "현재 노이즈가 얼마나 껴있는지"를 모델이 알 수 있게 합니다.
    """
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=32):
        super().__init__()
        # 첫 번째 block
        self.block1 = Block(dim, dim_out, groups=groups)
        # 두 번째 block
        self.block2 = Block(dim_out, dim_out, groups=groups)
        
        # 시간 정보를 Feature Map 채널 수에 맞게 변환하는 MLP
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
        else:
            self.mlp = None
            
        # channel 수가 다를 경우 residual 연결을 위해서 1x1 conv 사용
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        # 1. 첫번째 conv 통과
        h = self.block1(x)

        # 2. 시간 정보 주입 (time embedding injection)
        if self.mlp is not None and time_emb is not None:
            # (batch, time-dim) -> (batch, out_channel)
            time_emb = self.mlp(time_emb)
            # (batch, channel) -> (batch, channel, 1, 1)로 모양 맞추기
            # 이미지 전체 팍셀에 시간 정보를 더해줌(brodacasting add)
            h = h + time_emb[:, :, None, None]

        # 3. 두 번째 conv 통과        
        h = self.block2(h)

        # 4. residual connection
        return h + self.res_conv(x)

class Downsample(nn.Module):
    """
    [Downsampling] H, W를 절반으로 줄임
    Pooling 대신 Stride 2 Convolution을 사용하여 정보를 학습하며 줄임.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """
    [Upsampling] H, W를 2배로 늘림
    Transposed Conv 대신 'Nearest Interpolation + Conv'를 사용하여 
    Checkerboard Artifact(격자 무늬 노이즈)를 방지함.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        
    def forward(self, x):
        # Nearest Neighbor Interpolation으로 크기를 2배로 늘림
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def default(val, d):
    """
    유틸리티 함수: val이 None이면 d를 반환
    """
    if val is not None:
        return val
    return d


# ==============================================================================
# 2. Main U-Net Model (Phase 1 - No Attention)
# ==============================================================================

class Unet(nn.Module):
    """
    [DDPM Noise Prediction Model]
    입력: 노이즈 낀 이미지 (x_t), 타임스텝 (t)
    출력: 예측된 노이즈 (epsilon)
    구조: Encoder(Down) -> Bottleneck(Mid) -> Decoder(Up)
    """
    def __init__(
        self,
        dim=64,                # 기본 채널 수 (가장 얕은 곳)
        init_dim=None,         # 초기 Convolution 출력 채널
        out_dim=None,          # 최종 출력 채널 (RGB면 3)
        dim_mults=(1, 2, 2, 4),# 채널 배수 (예: 64 -> 128 -> 128 -> 256)
        channels=3,            # 입력 이미지 채널 (RGB)
        with_time_emb=True     # 시간 임베딩 사용 여부
    ):
        super().__init__()
        self.channels = channels
        
        init_dim = default(init_dim, dim)

        # 초기 이미지 특징 추출 (RGB -> Hidden Channels)
        # CIFAR-10 (32x32)에는 3x3 커널이 더 적합합니다. (7x7은 ImageNet용)
        self.init_conv = nn.Conv2d(channels, init_dim, 3, padding=1)
        #self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        # 각 단계별 channel 수 계산
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # e.g. [(64, 64), (64, 128), (128, 128), (128, 256)]

        # time embedding MLP 구축
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

        # ----------------------------------------------------------------------
        # 1. Encoder (Down Path) 구성
        # ----------------------------------------------------------------------
        skip_dims = []
        cur_dim = init_dim
        
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (num_resolutions - 1)
            
            # 각 레벨 : ResBlock -> ResBlock -> Downsample
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                # 마지막 레벨이 아니면 downsample 적용, 마지막 레벨은 identity
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))
            
            # Skip connection을 위한 현재 채널 수 기록
            skip_dims.append(dim_out)
            cur_dim = dim_out

        # ----------------------------------------------------------------------
        # 2. Bottleneck (Middle Path) 구성
        # 가장 깊은 곳에서 Global한 특징을 정제
        # Attention이 들어간다면 보통 여기에 추가됨
        # ----------------------------------------------------------------------
        self.mid_block1 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)
        self.mid_block2 = ResnetBlock(cur_dim, cur_dim, time_emb_dim=time_dim)

        # ----------------------------------------------------------------------
        # 3. Decoder (Up Path) 구성
        # ----------------------------------------------------------------------
        skip_dims_reversed = list(reversed(skip_dims))
        
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = i >= (num_resolutions - 1)
            # encoder에서 가져올 skip connection의 채널 수
            skip_dim = skip_dims_reversed[i]
            
            # decoder의 입력 채널 = (이전 레이어 출력) + (skip connection)
            # dim_out: 이전 layer에서 올라온 channel(변수명이 reversed라 반대임을 주의)
            # skip_dim: encoder에서 가져온 skip connection의 channel
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + skip_dim, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                # 마지막 level이 아니면 upsample 적용, 마지막 level은 identity
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
            
        self.out_dim = default(out_dim, channels)
        # ----------------------------------------------------------------------
        # 4. Final Projection
        # ----------------------------------------------------------------------
        self.final_res_block = ResnetBlock(init_dim, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)


    def forward(self, x, time):
        # 1. initial convolution
        x = self.init_conv(x)
        r = x.clone()
        
        # 2. time embedding
        t = self.time_mlp(time) if self.time_mlp is not None else None
        
        # 3. down path(encoder)
        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h.append(x) # Skip connection 저장(현재 resolution의 feature map)
            x = downsample(x)
            
        # 4. Mid path(bottleneck)
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # 5. Up path(decoder)
        for block1, block2, upsample in self.ups:
            # encoder에서 저장해준 feature map 가져오기(pop)
            skip = h.pop()
            # channel dimension(dim=1)로 결합(concatenate)
            x = torch.cat((x, skip), dim=1)
            
            x = block1(x, t)
            x = block2(x, t)
            x = upsample(x)
            
        # 6. Final output (noise prediction)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
