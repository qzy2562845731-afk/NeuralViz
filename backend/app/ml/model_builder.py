"""
模型构建器
根据数据集类型与特征维度，动态构建可训练的 PyTorch 模型
- 图像数据集（image_folder / mnist_idx）：构建 CNN（支持 8 种注意力机制、可配置激活函数）
- 表格数据集（csv / numpy）：构建 MLP（支持 SE-like 通道注意力）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from app.ml.attention import create_attention, ATTENTION_REGISTRY


def _parse_image_shape(feature_shape) -> Tuple[int, int, int]:
    """解析图像特征形状为 (channels, height, width)
    
    支持的格式：
    - list/tuple [C, H, W] 或 [H, W]
    - 字符串 "CxHxW"（NCHW格式，如 "1x28x28"、"3x32x32"）
    - 字符串 "HxW"（默认单通道）
    """
    if isinstance(feature_shape, (list, tuple)):
        if len(feature_shape) == 3:
            return int(feature_shape[0]), int(feature_shape[1]), int(feature_shape[2])
        elif len(feature_shape) == 2:
            return 1, int(feature_shape[0]), int(feature_shape[1])
        return 1, 28, 28
    if not isinstance(feature_shape, str):
        return 1, 28, 28
    try:
        parts = feature_shape.lower().split("x")
        if len(parts) == 3:
            c, h, w = int(parts[0]), int(parts[1]), int(parts[2])
            if c <= 4 and h > 4 and w > 4:
                return c, h, w
            if w <= 4 and h > 4 and c > 4:
                return w, h, c
            return c, h, w
        elif len(parts) == 2:
            h, w = int(parts[0]), int(parts[1])
            return 1, h, w
    except (ValueError, IndexError):
        pass
    return 1, 28, 28


def _parse_feature_dim(feature_shape: str) -> int:
    try:
        return int(feature_shape.strip())
    except (ValueError, AttributeError):
        return 784


# ============================================================
# 注意力机制模块（统一从 app.ml.attention 导入）
# ============================================================

from app.ml.attention import (
    SEBlock, CBAMBlock, SelfAttention2d,
    ECA, MHSA2d, GCT, CoordAttention,
    create_attention, ATTENTION_REGISTRY,
)

# 保留旧 API 兼容别名
ChannelAttention = SEBlock  # 向后兼容
SpatialAttention = None     # 已内聚到 CBAMBlock 中


# ============================================================
# 可配置 CNN 模型
# ============================================================

class ConfigurableCNN(nn.Module):
    """可配置图像分类 CNN

    支持：
    - 自定义通道数列表（如 [32, 64, 128]）
    - 8 种注意力机制（none/se/eca/cbam/self_attention/mhsa/gct/coord）
    - 可配置激活函数（relu/leaky_relu/gelu/silu/tanh）
    - 消融实验：可移除注意力/残差连接/批归一化/Dropout 等组件
    """

    # 支持的激活函数注册表
    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
    }

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channel_list: Optional[List[int]] = None,
        attention: str = "none",
        use_bn: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
        use_residual: bool = False,
        fc_hidden: int = 128,
        activation: str = "relu",
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if channel_list is None:
            channel_list = [32, 64]

        # 输入验证
        if not isinstance(channel_list, (list, tuple)) or len(channel_list) == 0:
            raise ValueError(f"channel_list 必须是非空列表，当前值: {channel_list}")
        for i, ch in enumerate(channel_list):
            if not isinstance(ch, int) or ch <= 0:
                raise ValueError(f"channel_list[{i}] 必须是正整数，当前值: {ch}")
        if in_channels <= 0:
            raise ValueError(f"in_channels 必须是正整数，当前值: {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes 必须是正整数，当前值: {num_classes}")

        self.channel_list = list(channel_list)
        self.attention_type = attention
        self.use_residual = use_residual

        # 激活函数解析
        act_cls = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)
        attn_kwargs = attention_kwargs or {}

        # 构建卷积层
        layers = []
        in_ch = in_channels
        self.attn_modules = nn.ModuleDict()
        self.bn_modules = nn.ModuleDict()
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self._residual_projs = nn.ModuleDict()

        for i, out_ch in enumerate(channel_list):
            block = nn.Sequential()
            block.add_module(f"conv{i}", nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                block.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            # 使用可配置激活函数
            if activation == "leaky_relu":
                block.add_module(f"act{i}", act_cls(negative_slope=0.1, inplace=True))
            elif activation in ("relu", "gelu", "silu"):
                block.add_module(f"act{i}", act_cls(inplace=True))
            else:
                block.add_module(f"act{i}", act_cls())
            self.conv_layers.append(block)
            self.pool_layers.append(nn.MaxPool2d(2))

            # 注意力模块（使用工厂函数）
            attn_module = create_attention(attention, out_ch, **attn_kwargs)
            if attn_module is not None:
                self.attn_modules[f"attn{i}"] = attn_module

            # 残差连接的 1x1 卷积（通道数变化时）
            if use_residual and in_ch != out_ch:
                self._make_residual_proj(i, in_ch, out_ch)
            in_ch = out_ch

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        pool_dim = channel_list[-1] * 4 * 4

        # 分类头
        classifier_layers = [nn.Flatten()]
        classifier_layers.append(nn.Linear(pool_dim, fc_hidden))
        classifier_layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(fc_hidden, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _make_residual_proj(self, idx: int, in_ch: int, out_ch: int):
        self._residual_projs[f"proj{idx}"] = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.conv_layers):
            identity = x
            x = conv(x)
            x = self.pool_layers[i](x)
            # 注意力
            attn_name = f"attn{i}"
            if attn_name in self.attn_modules:
                x = self.attn_modules[attn_name](x)
            # 残差
            if self.use_residual:
                proj_name = f"proj{i}"
                if proj_name in self._residual_projs:
                    identity = self._residual_projs[proj_name](identity)
                # 对 identity 做池化以匹配尺寸
                if identity.shape[2:] != x.shape[2:]:
                    identity = F.adaptive_avg_pool2d(identity, x.shape[2:])
                x = x + identity
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取各层特征图用于可视化"""
        features = {}
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.pool_layers[i](x)
            attn_name = f"attn{i}"
            if attn_name in self.attn_modules:
                x = self.attn_modules[attn_name](x)
            features[f"conv{i+1}"] = x.detach()
        return features

    def get_kernels(self) -> Dict[str, torch.Tensor]:
        """提取第一层卷积核用于可视化"""
        kernels = {}
        for i, conv in enumerate(self.conv_layers):
            for name, param in conv.named_parameters():
                if "weight" in name and param.dim() == 4:
                    kernels[f"conv{i+1}_weight"] = param.detach()
                    break
        return kernels


class SimpleCNN(ConfigurableCNN):
    """兼容旧代码：默认 SimpleCNN = ConfigurableCNN(channel_list=[32,64])"""
    def __init__(self, in_channels: int, num_classes: int, input_size: int = 28, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=num_classes,
                         channel_list=[32, 64], attention="none",
                         use_bn=True, use_dropout=True, **kwargs)


# ============================================================
# 可配置 MLP（支持通道注意力式特征重标定）
# ============================================================

class ConfigurableMLP(nn.Module):
    """可配置表格分类 MLP

    支持：自定义隐藏层维度、SE-like特征重标定、Dropout
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        use_bn: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            h1 = max(64, min(in_features, 256))
            h2 = max(32, h1 // 2)
            hidden_dims = [h1, h2]

        # 输入验证
        if in_features <= 0:
            raise ValueError(f"in_features 必须是正整数，当前值: {in_features}")
        if num_classes <= 0:
            raise ValueError(f"num_classes 必须是正整数，当前值: {num_classes}")
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
            raise ValueError(f"hidden_dims 必须是非空列表，当前值: {hidden_dims}")
        for i, dim in enumerate(hidden_dims):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"hidden_dims[{i}] 必须是正整数，当前值: {dim}")

        layers = []
        prev_dim = in_features
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

        # SE-like feature reweighting for MLP
        self.use_attention = use_attention
        if use_attention:
            mid = max(in_features // 16, 4)
            self.se = nn.Sequential(
                nn.Linear(in_features, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, in_features, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            w = self.se(x)
            x = x * w
        return self.net(x)


class SimpleMLP(ConfigurableMLP):
    """兼容旧代码"""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__(in_features=in_features, num_classes=num_classes,
                         use_bn=False, use_dropout=True, use_attention=False)


# ============================================================
# 现代模型架构
# ============================================================

class BasicBlock(nn.Module):
    """ResNet BasicBlock: 两个 3x3 卷积 + 残差连接"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, use_bn: bool = True,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 实现

    标准 ResNet-18 架构，支持自定义输入通道和类别数
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 use_bn: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, use_bn=use_bn, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, use_bn=use_bn, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int,
                    stride: int = 1, use_bn: bool = True, dropout_rate: float = 0.0):
        downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=not use_bn),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion) if use_bn else nn.Identity(),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample, use_bn, dropout_rate))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels * BasicBlock.expansion, out_channels, use_bn=use_bn, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual Block with SE"""

    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 expand_ratio: int, use_se: bool = True, use_bn: bool = True):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=not use_bn))
            layers.append(nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity())
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                      groups=hidden_dim, bias=not use_bn),
            nn.BatchNorm2d(hidden_dim) if use_bn else nn.Identity(),
            nn.ReLU6(inplace=True),
        ])

        if use_se:
            se_reduction = max(1, hidden_dim // 4)
            layers.append(SEBlock(hidden_dim, reduction=se_reduction))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3(nn.Module):
    """MobileNetV3-Small 实现

    轻量级移动端CNN，适合资源受限场景
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 use_bn: bool = True, dropout_rate: float = 0.2):
        super().__init__()
        # 配置: [in_c, out_c, stride, expand_ratio, use_se]
        configs = [
            [16, 16, 2, 1, True],
            [16, 24, 2, 4, False],
            [24, 24, 1, 3, False],
            [24, 40, 2, 3, True],
            [40, 40, 1, 3, True],
            [40, 40, 1, 3, True],
            [40, 48, 1, 3, True],
            [48, 48, 1, 3, True],
            [48, 96, 2, 6, True],
            [96, 96, 1, 6, True],
            [96, 96, 1, 6, True],
        ]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=not use_bn),
            nn.BatchNorm2d(16) if use_bn else nn.Identity(),
            nn.ReLU6(inplace=True),
        )

        layers = []
        for in_c, out_c, stride, expand, use_se in configs:
            layers.append(InvertedResidual(in_c, out_c, stride, expand, use_se=use_se, use_bn=use_bn))

        self.features = nn.Sequential(*layers)

        last_conv_in = 96
        last_conv_out = 576
        self.last_conv = nn.Sequential(
            nn.Conv2d(last_conv_in, last_conv_out, kernel_size=1, bias=not use_bn),
            nn.BatchNorm2d(last_conv_out) if use_bn else nn.Identity(),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(last_conv_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PatchEmbedding(nn.Module):
    """ViT Patch Embedding"""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, E)
        return x


class ViT(nn.Module):
    """Vision Transformer (ViT) 简化实现

    适用于小尺寸图像（如 MNIST 28x28, CIFAR 32x32）
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 img_size: int = 28, patch_size: int = 4, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, mlp_ratio: int = 4,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        return x


# ============================================================
# 模型架构注册表
# ============================================================

MODEL_ARCHITECTURES = {
    "cnn": "ConfigurableCNN",
    "mlp": "ConfigurableMLP",
    "resnet18": "ResNet18",
    "mobilenetv3": "MobileNetV3",
    "vit": "ViT",
}


def build_model_by_architecture(
    architecture: str,
    in_channels: int,
    num_classes: int,
    img_size: int = 28,
    model_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """根据架构名称构建模型

    Args:
        architecture: 架构名称 (cnn/mlp/resnet18/mobilenetv3/vit)
        in_channels: 输入通道数
        num_classes: 类别数
        img_size: 图像尺寸（用于 ViT）
        model_config: 额外配置

    Returns:
        PyTorch 模型
    """
    cfg = model_config or {}

    if architecture == "resnet18":
        return ResNet18(
            in_channels=in_channels,
            num_classes=num_classes,
            use_bn=cfg.get("use_bn", True),
            dropout_rate=cfg.get("dropout_rate", 0.0),
        )
    elif architecture == "mobilenetv3":
        return MobileNetV3(
            in_channels=in_channels,
            num_classes=num_classes,
            use_bn=cfg.get("use_bn", True),
            dropout_rate=cfg.get("dropout_rate", 0.2),
        )
    elif architecture == "vit":
        return ViT(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            patch_size=cfg.get("patch_size", 4),
            embed_dim=cfg.get("embed_dim", 128),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 4),
            dropout_rate=cfg.get("dropout_rate", 0.1),
        )
    elif architecture == "mlp":
        from app.ml.model_builder import _parse_feature_dim
        in_features = in_channels * img_size * img_size
        return ConfigurableMLP(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=cfg.get("hidden_dims"),
            use_bn=cfg.get("use_bn", False),
            use_dropout=cfg.get("use_dropout", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            use_attention=cfg.get("use_attention", False),
        )
    else:  # 默认 CNN
        return ConfigurableCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_list=cfg.get("channels", [32, 64]),
            attention=cfg.get("attention", "none"),
            use_bn=cfg.get("use_bn", True),
            use_dropout=cfg.get("use_dropout", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            use_residual=cfg.get("use_residual", False),
            fc_hidden=cfg.get("fc_hidden", 128),
            activation=cfg.get("activation", "relu"),
            attention_kwargs=cfg.get("attention_kwargs"),
        )


# ============================================================
# 模型构建入口
# ============================================================

def build_model(
    dataset_type: str,
    feature_shape: str,
    num_classes: int,
    model_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """根据数据集类型与特征维度构建 PyTorch 模型

    Args:
        dataset_type: 数据集格式标识 (image_folder / mnist_idx / csv / numpy)
        feature_shape: 特征维度字符串
        num_classes: 类别数
        model_config: 可选模型配置字典，支持字段：
            - channels: List[int] 卷积通道数列表，默认 [32,64]
            - attention: str 注意力类型，支持 none/se/eca/cbam/self_attention/mhsa/gct/coord
            - activation: str 激活函数，支持 relu/leaky_relu/gelu/silu/tanh
            - attention_kwargs: Dict 注意力额外参数
            - use_bn: bool 是否使用批归一化
            - use_dropout: bool 是否使用Dropout
            - dropout_rate: float Dropout比例
            - use_residual: bool 是否使用残差连接
            - fc_hidden: int 全连接层隐藏维度
            - hidden_dims: List[int] MLP隐藏层维度

    Returns:
        可训练的 torch.nn.Module
    """
    cfg = model_config or {}
    if dataset_type in ("image_folder", "mnist_idx"):
        c, h, w = _parse_image_shape(feature_shape)
        return ConfigurableCNN(
            in_channels=c,
            num_classes=num_classes,
            channel_list=cfg.get("channels", [32, 64]),
            attention=cfg.get("attention", "none"),
            use_bn=cfg.get("use_bn", True),
            use_dropout=cfg.get("use_dropout", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            use_residual=cfg.get("use_residual", False),
            fc_hidden=cfg.get("fc_hidden", 128),
            activation=cfg.get("activation", "relu"),
            attention_kwargs=cfg.get("attention_kwargs"),
        )
    else:
        in_features = _parse_feature_dim(feature_shape)
        return ConfigurableMLP(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=cfg.get("hidden_dims"),
            use_bn=cfg.get("use_bn", False),
            use_dropout=cfg.get("use_dropout", True),
            dropout_rate=cfg.get("dropout_rate", 0.3),
            use_attention=cfg.get("use_attention", False),
        )


def load_trained_model(
    model_path: str,
    dataset_type: str,
    feature_shape: str,
    num_classes: int,
    model_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    model = build_model(dataset_type, feature_shape, num_classes, model_config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
