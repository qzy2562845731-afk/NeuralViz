"""
模型构建器单元测试
覆盖: ConfigurableCNN, ConfigurableMLP, build_model, count_parameters, _parse_image_shape
"""
import pytest
import torch
from app.ml.model_builder import (
    build_model, count_parameters, ConfigurableCNN, ConfigurableMLP,
    SEBlock, CBAMBlock, SelfAttention2d, SimpleCNN, SimpleMLP,
    _parse_image_shape, load_trained_model,
)


class TestParseImageShape:
    """测试图像形状解析"""

    def test_parse_nchw_string(self):
        """解析 NCHW 格式字符串"""
        c, h, w = _parse_image_shape("1x28x28")
        assert c == 1
        assert h == 28
        assert w == 28

    def test_parse_hwc_string(self):
        """解析 HWC 格式字符串"""
        c, h, w = _parse_image_shape("28x28x1")
        assert c == 1
        assert h == 28
        assert w == 28

    def test_parse_rgb_nchw(self):
        """解析 RGB NCHW 格式"""
        c, h, w = _parse_image_shape("3x32x32")
        assert c == 3
        assert h == 32
        assert w == 32

    def test_parse_hw_string(self):
        """解析 HW 格式"""
        c, h, w = _parse_image_shape("28x28")
        assert c == 1
        assert h == 28
        assert w == 28

    def test_parse_list_format(self):
        """解析 list 格式"""
        c, h, w = _parse_image_shape([1, 28, 28])
        assert c == 1
        assert h == 28
        assert w == 28

    def test_parse_invalid_format(self):
        """解析无效格式返回默认值"""
        c, h, w = _parse_image_shape("invalid")
        assert c == 1
        assert h == 28
        assert w == 28

    def test_parse_non_string(self):
        """解析非字符串"""
        c, h, w = _parse_image_shape(784)
        assert c == 1
        assert h == 28
        assert w == 28


class TestConfigurableCNN:
    """测试可配置 CNN 模型"""

    def test_basic_cnn_creation(self):
        """基础 CNN 创建"""
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        assert isinstance(model, ConfigurableCNN)
        assert model.channel_list == [32, 64]

    def test_cnn_forward_pass(self, sample_image_tensors):
        """CNN 前向传播"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_with_se_attention(self, sample_image_tensors):
        """CNN + SE 注意力"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, attention="se")
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_with_cbam_attention(self, sample_image_tensors):
        """CNN + CBAM 注意力"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, attention="cbam")
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_with_self_attention(self, sample_image_tensors):
        """CNN + Self-Attention"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, attention="self_attention")
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_with_residual(self, sample_image_tensors):
        """CNN + 残差连接"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, use_residual=True)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_without_bn(self, sample_image_tensors):
        """CNN 无 BatchNorm"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, use_bn=False)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_without_dropout(self, sample_image_tensors):
        """CNN 无 Dropout"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, use_dropout=False)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_custom_channels(self, sample_image_tensors):
        """CNN 自定义通道数"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10, channel_list=[16, 32, 64])
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)

    def test_cnn_rgb_input(self):
        """CNN RGB 输入"""
        model = ConfigurableCNN(in_channels=3, num_classes=10)
        X = torch.randn(4, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            output = model(X)
        assert output.shape == (4, 10)

    def test_get_feature_maps(self, sample_image_tensors):
        """获取特征图"""
        X, _ = sample_image_tensors
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        fm = model.get_feature_maps(X[:1])
        assert len(fm) > 0
        for name, tensor in fm.items():
            assert tensor.dim() == 4  # (1, C, H, W)

    def test_get_kernels(self):
        """获取卷积核"""
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        kernels = model.get_kernels()
        assert len(kernels) > 0

    def test_simple_cnn_compatibility(self, sample_image_tensors):
        """SimpleCNN 兼容性"""
        X, _ = sample_image_tensors
        model = SimpleCNN(in_channels=1, num_classes=10)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 10)


class TestConfigurableMLP:
    """测试可配置 MLP 模型"""

    def test_basic_mlp_creation(self):
        """基础 MLP 创建"""
        model = ConfigurableMLP(in_features=784, num_classes=10)
        assert isinstance(model, ConfigurableMLP)

    def test_mlp_forward_pass(self, sample_tabular_tensors):
        """MLP 前向传播"""
        X, _ = sample_tabular_tensors
        model = ConfigurableMLP(in_features=64, num_classes=5)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 5)

    def test_mlp_with_attention(self, sample_tabular_tensors):
        """MLP + SE-like 注意力"""
        X, _ = sample_tabular_tensors
        model = ConfigurableMLP(in_features=64, num_classes=5, use_attention=True)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 5)

    def test_mlp_without_bn(self, sample_tabular_tensors):
        """MLP 无 BatchNorm"""
        X, _ = sample_tabular_tensors
        model = ConfigurableMLP(in_features=64, num_classes=5, use_bn=False)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 5)

    def test_mlp_custom_hidden_dims(self, sample_tabular_tensors):
        """MLP 自定义隐藏层"""
        X, _ = sample_tabular_tensors
        model = ConfigurableMLP(in_features=64, num_classes=5, hidden_dims=[128, 64, 32])
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 5)

    def test_simple_mlp_compatibility(self, sample_tabular_tensors):
        """SimpleMLP 兼容性"""
        X, _ = sample_tabular_tensors
        model = SimpleMLP(in_features=64, num_classes=5)
        model.eval()
        with torch.no_grad():
            output = model(X[:4])
        assert output.shape == (4, 5)


class TestAttentionModules:
    """测试注意力模块"""

    def test_se_block(self, sample_image_tensors):
        """SE Block"""
        X, _ = sample_image_tensors
        se = SEBlock(channels=32)
        se.eval()
        with torch.no_grad():
            output = se(X[:4, :1, :, :].expand(-1, 32, -1, -1))
        assert output.shape == (4, 32, 28, 28)

    def test_cbam_block(self, sample_image_tensors):
        """CBAM Block"""
        X, _ = sample_image_tensors
        cbam = CBAMBlock(channels=32)
        cbam.eval()
        with torch.no_grad():
            output = cbam(X[:4, :1, :, :].expand(-1, 32, -1, -1))
        assert output.shape == (4, 32, 28, 28)

    def test_self_attention_2d(self, sample_image_tensors):
        """Self-Attention 2D"""
        X, _ = sample_image_tensors
        sa = SelfAttention2d(in_channels=32)
        sa.eval()
        with torch.no_grad():
            output = sa(X[:4, :1, :, :].expand(-1, 32, -1, -1))
        assert output.shape == (4, 32, 28, 28)


class TestBuildModel:
    """测试 build_model 工厂函数"""

    def test_build_image_model(self):
        """构建图像模型"""
        model = build_model("mnist_idx", "1x28x28", 10, {"channels": [32, 64]})
        assert isinstance(model, ConfigurableCNN)

    def test_build_tabular_model(self):
        """构建表格模型"""
        model = build_model("csv", "64", 5)
        assert isinstance(model, ConfigurableMLP)

    def test_build_numpy_model(self):
        """构建 numpy 数据集模型"""
        model = build_model("numpy", "128", 3)
        assert isinstance(model, ConfigurableMLP)

    def test_build_with_config(self):
        """带配置构建模型"""
        config = {
            "channels": [32, 64, 128],
            "attention": "se",
            "use_bn": True,
            "use_dropout": True,
            "dropout_rate": 0.5,
            "fc_hidden": 256,
        }
        model = build_model("image_folder", "3x32x32", 10, config)
        assert isinstance(model, ConfigurableCNN)
        assert model.channel_list == [32, 64, 128]
        assert model.attention_type == "se"


class TestCountParameters:
    """测试参数量统计"""

    def test_cnn_params(self):
        """CNN 参数量"""
        model = ConfigurableCNN(in_channels=1, num_classes=10)
        params = count_parameters(model)
        assert params > 0
        assert isinstance(params, int)

    def test_mlp_params(self):
        """MLP 参数量"""
        model = ConfigurableMLP(in_features=784, num_classes=10)
        params = count_parameters(model)
        assert params > 0
        assert isinstance(params, int)

    def test_different_architectures_have_different_params(self):
        """不同架构参数量不同"""
        cnn = ConfigurableCNN(in_channels=1, num_classes=10, channel_list=[32, 64])
        cnn_big = ConfigurableCNN(in_channels=1, num_classes=10, channel_list=[32, 64, 128, 256])
        assert count_parameters(cnn) < count_parameters(cnn_big)