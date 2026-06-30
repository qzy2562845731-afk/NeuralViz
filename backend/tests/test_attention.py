"""
注意力机制单元测试
覆盖: SEBlock, ECA, CBAMBlock, SelfAttention2d, MHSA2d, GCT, CoordAttention, create_attention
"""
import pytest
import torch
import torch.nn as nn
from app.ml.attention import (
    SEBlock, ECA, CBAMBlock, SelfAttention2d, MHSA2d, GCT, CoordAttention,
    create_attention, ATTENTION_REGISTRY,
)


class TestSEBlock:
    """SE (Squeeze-and-Excitation) 通道注意力"""

    def test_creation(self):
        block = SEBlock(channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = SEBlock(channels=32)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)

    def test_with_reduction(self):
        block = SEBlock(channels=64, reduction=8)
        x = torch.randn(2, 64, 8, 8)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 8, 8)


class TestECA:
    """ECA (Efficient Channel Attention)"""

    def test_creation(self):
        block = ECA(channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = ECA(channels=32, k_size=3)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)

    def test_auto_kernel_size(self):
        """自适应 kernel size"""
        block = ECA(channels=128)
        x = torch.randn(2, 128, 8, 8)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 128, 8, 8)


class TestCBAMBlock:
    """CBAM (Convolutional Block Attention Module)"""

    def test_creation(self):
        block = CBAMBlock(channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = CBAMBlock(channels=32)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)

    def test_with_reduction(self):
        block = CBAMBlock(channels=64, reduction=8)
        x = torch.randn(2, 64, 8, 8)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 8, 8)


class TestSelfAttention2d:
    """Self-Attention 2D (Non-local)"""

    def test_creation(self):
        block = SelfAttention2d(in_channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = SelfAttention2d(in_channels=32)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)

    def test_with_reduction(self):
        block = SelfAttention2d(in_channels=64, reduction=4)
        x = torch.randn(2, 64, 8, 8)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 64, 8, 8)


class TestMHSA2d:
    """Multi-Head Self-Attention 2D"""

    def test_creation(self):
        block = MHSA2d(in_channels=64, num_heads=4, head_dim=32)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = MHSA2d(in_channels=32, num_heads=4, head_dim=32)
        x = torch.randn(2, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 32, 16, 16)

    def test_with_dropout(self):
        block = MHSA2d(in_channels=32, num_heads=4, head_dim=32, dropout=0.1)
        x = torch.randn(2, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 32, 16, 16)


class TestGCT:
    """GCT (Gated Channel Transformation)"""

    def test_creation(self):
        block = GCT(channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = GCT(channels=32)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)


class TestCoordAttention:
    """Coordinate Attention"""

    def test_creation(self):
        block = CoordAttention(channels=64)
        assert isinstance(block, nn.Module)

    def test_forward(self):
        block = CoordAttention(channels=32)
        x = torch.randn(4, 32, 16, 16)
        block.eval()
        with torch.no_grad():
            out = block(x)
        assert out.shape == (4, 32, 16, 16)


class TestCreateAttention:
    """注意力工厂函数"""

    def test_create_none(self):
        """创建 'none' 应返回 None"""
        result = create_attention("none", 32)
        assert result is None

    def test_create_se(self):
        attn = create_attention("se", 64)
        assert isinstance(attn, SEBlock)

    def test_create_eca(self):
        attn = create_attention("eca", 64, eca_kernel_size=5)
        assert isinstance(attn, ECA)

    def test_create_cbam(self):
        attn = create_attention("cbam", 64, reduction=8)
        assert isinstance(attn, CBAMBlock)

    def test_create_self_attention(self):
        attn = create_attention("self_attention", 64)
        assert isinstance(attn, SelfAttention2d)

    def test_create_mhsa(self):
        attn = create_attention("mhsa", 64, num_heads=8, head_dim=32)
        assert isinstance(attn, MHSA2d)

    def test_create_gct(self):
        attn = create_attention("gct", 64)
        assert isinstance(attn, GCT)

    def test_create_coord(self):
        attn = create_attention("coord", 64)
        assert isinstance(attn, CoordAttention)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="不支持的注意力类型"):
            create_attention("invalid_attention", 32)

    def test_registry_contains_all_types(self):
        """注册表包含所有 8 种注意力类型"""
        expected = {"none", "se", "eca", "cbam", "self_attention", "mhsa", "gct", "coord"}
        assert set(ATTENTION_REGISTRY.keys()) == expected

    def test_all_attention_types_forward(self):
        """所有注意力类型的前向传播不报错"""
        x = torch.randn(2, 32, 8, 8)
        for attn_type in ATTENTION_REGISTRY:
            if attn_type == "none":
                continue
            attn = create_attention(attn_type, 32)
            if attn is None:
                continue
            attn.eval()
            with torch.no_grad():
                out = attn(x)
            assert out.shape == x.shape, f"{attn_type} 输出形状不匹配"