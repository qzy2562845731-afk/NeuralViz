"""
注意力机制模块
提供多种注意力机制，支持即插即用到 CNN 模型中：
- SE: Squeeze-and-Excitation 通道注意力
- CBAM: Convolutional Block Attention Module (通道+空间)
- SelfAttention2d: 自注意力 (Non-local)
- ECA: Efficient Channel Attention (轻量级通道注意力)
- MHSA: Multi-Head Self-Attention for 2D feature maps
- GCT: Gated Channel Transformation (门控通道变换)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 基础注意力模块
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (通道注意力)
    arXiv: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = max(1, reduction)
        mid = max(channels // self.reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    """Efficient Channel Attention (轻量级通道注意力)
    使用 1D 卷积替代全连接层，参数量极低
    arXiv: https://arxiv.org/abs/1910.03151
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应卷积核大小: k = |log2(C)/γ + b/γ|_odd
        if k_size is None:
            t = int(abs((math.log2(channels) + 1) / 2))
            k_size = t if t % 2 == 1 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (通道+空间注意力)
    arXiv: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid_ch = nn.Sigmoid()
        padding = kernel_size // 2
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        ch_attn = self.sigmoid_ch(avg_out + max_out).view(b, c, 1, 1)
        x = x * ch_attn
        # 空间注意力
        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = self.sigmoid_sp(self.conv_spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x * sp_attn


class SelfAttention2d(nn.Module):
    """自注意力模块 (Non-local / Self-Attention for 2D feature maps)
    通过 Query/Key/Value 计算空间位置间的长程依赖
    """
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.query = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.key = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)
        energy = torch.bmm(q, k)
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


# ============================================================
# 高级注意力模块
# ============================================================

class MHSA2d(nn.Module):
    """Multi-Head Self-Attention for 2D feature maps
    将特征图按 head 分割，在多个子空间中并行计算自注意力
    支持 position encoding 和 dropout
    """
    def __init__(self, in_channels: int, num_heads: int = 4, head_dim: int = 32,
                 dropout: float = 0.0, use_pos_encoding: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.use_pos_encoding = use_pos_encoding

        self.qkv = nn.Conv2d(in_channels, self.inner_dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, self.inner_dim, 32, 32) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        qkv = self.qkv(x).view(b, 3, self.num_heads, self.head_dim, h, w)
        q, k, v = qkv.unbind(1)  # each: (B, heads, head_dim, h, w)

        # 添加位置编码
        if self.use_pos_encoding and self.pos_encoding is not None:
            pe = F.interpolate(self.pos_encoding, size=(h, w), mode='bilinear', align_corners=False)
            pe = pe.view(1, self.num_heads, self.head_dim, h, w)
            q = q + pe
            k = k + pe

        # 变形为 (B, heads, head_dim, N)
        q = q.view(b, self.num_heads, self.head_dim, -1)
        k = k.view(b, self.num_heads, self.head_dim, -1)
        v = v.view(b, self.num_heads, self.head_dim, -1)

        # 计算注意力: (B, heads, N, N)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权求和: (B, heads, head_dim, N)
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, self.inner_dim, h, w)
        out = self.proj(out)
        out = self.dropout(out)
        return self.gamma * out + x


class GCT(nn.Module):
    """Gated Channel Transformation (门控通道变换)
    通过门控机制学习通道级特征重标定，比 SE 更轻量
    arXiv: https://arxiv.org/abs/1909.11519
    """
    def __init__(self, channels: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm per channel
        embedding = (x.pow(2).sum(dim=(2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        # 通道归一化
        norm = embedding / (embedding.mean(dim=1, keepdim=True) + self.epsilon)
        # 门控
        gate = 1.0 + torch.tanh(self.gamma * norm + self.beta)
        return x * gate


class CoordAttention(nn.Module):
    """Coordinate Attention (坐标注意力)
    沿水平和垂直方向分别编码空间信息，保留精确位置信息
    arXiv: https://arxiv.org/abs/2103.02907
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # 水平池化: (B, C, H, 1)
        x_h = self.pool_h(x)
        # 垂直池化: (B, C, 1, W)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 共享卷积
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 分离注意力权重
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


# ============================================================
# 注意力工厂函数
# ============================================================

ATTENTION_REGISTRY = {
    "none": None,
    "se": SEBlock,
    "eca": ECA,
    "cbam": CBAMBlock,
    "self_attention": SelfAttention2d,
    "mhsa": MHSA2d,
    "gct": GCT,
    "coord": CoordAttention,
}


def create_attention(attention_type: str, channels: int, **kwargs) -> nn.Module:
    """注意力模块工厂函数

    Args:
        attention_type: 注意力类型
            - none: 无注意力
            - se: Squeeze-and-Excitation
            - eca: Efficient Channel Attention
            - cbam: CBAM (通道+空间)
            - self_attention: Self-Attention (Non-local)
            - mhsa: Multi-Head Self-Attention
            - gct: Gated Channel Transformation
            - coord: Coordinate Attention
        channels: 输入通道数
        **kwargs: 额外参数（如 num_heads, reduction 等）

    Returns:
        注意力模块实例，若类型为 none 返回 None
    """
    if attention_type == "none" or attention_type is None:
        return None

    # 防御性处理：布尔值或其他非字符串类型
    if isinstance(attention_type, bool):
        attention_type = "se" if attention_type else "none"
        if attention_type == "none":
            return None
    elif not isinstance(attention_type, str):
        attention_type = str(attention_type)

    cls = ATTENTION_REGISTRY.get(attention_type)
    if cls is None:
        raise ValueError(f"不支持的注意力类型: {attention_type}，支持: {list(ATTENTION_REGISTRY.keys())}")

    # 根据类型传递合适参数
    if attention_type == "mhsa":
        return cls(channels, num_heads=kwargs.get("num_heads", 4),
                   head_dim=kwargs.get("head_dim", 32),
                   dropout=kwargs.get("attn_dropout", 0.0))
    elif attention_type == "eca":
        return cls(channels, k_size=kwargs.get("eca_kernel_size", 3))
    elif attention_type in ("se", "cbam"):
        return cls(channels, reduction=kwargs.get("reduction", 16))
    elif attention_type == "self_attention":
        return cls(channels, reduction=kwargs.get("reduction", 8))
    else:
        return cls(channels)