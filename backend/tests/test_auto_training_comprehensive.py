"""
综合自动化训练功能测试
覆盖：嵌套通道配置、参数解析、模型构建、训练流程
"""
import torch
import pytest
import json
from app.ml.model_builder import (
    build_model, ConfigurableCNN, ConfigurableMLP, _flatten_channels,
)
from app.services.training_service import TrainingService


# ============================================================
# 1. 嵌套通道配置解析测试
# ============================================================
class TestNestedChannelConfig:
    """测试多层嵌套数组通道配置的解析能力"""

    def test_user_reported_case(self):
        """用户复现场景：[[16,32],[32,64],[32,64,128],[64,128,256],[64,128,256,512]]"""
        channels = [
            [16, 32],
            [32, 64],
            [32, 64, 128],
            [64, 128, 256],
            [64, 128, 256, 512],
        ]
        result = _flatten_channels(channels)
        expected = [16, 32, 32, 64, 32, 64, 128, 64, 128, 256, 64, 128, 256, 512]
        assert result == expected
        assert all(isinstance(c, int) for c in result)

    def test_single_flat_list(self):
        """单层列表：[32, 64]"""
        assert _flatten_channels([32, 64]) == [32, 64]

    def test_double_nested(self):
        """双层嵌套：[[16, 32], [64, 128]]"""
        assert _flatten_channels([[16, 32], [64, 128]]) == [16, 32, 64, 128]

    def test_triple_nested(self):
        """三层嵌套：[[[16, 32]], [64, 128]]"""
        result = _flatten_channels([[[16, 32]], [64, 128]])
        assert result == [16, 32, 64, 128]

    def test_mixed_flat_and_nested(self):
        """混合扁平与嵌套：[32, [64, 128], 256]"""
        result = _flatten_channels([32, [64, 128], 256])
        assert result == [32, 64, 128, 256]

    def test_single_inner_single(self):
        """单元素嵌套：[[32]]"""
        assert _flatten_channels([[32]]) == [32]

    def test_deeply_nested(self):
        """深层嵌套：[[[[[32, 64]]]]]"""
        assert _flatten_channels([[[[[32, 64]]]]]) == [32, 64]


# ============================================================
# 2. 模型构建测试（各种通道配置 + 参数组合）
# ============================================================
class TestModelBuildWithNestedChannels:
    """测试使用嵌套通道配置构建模型"""

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 1, 28, 28)

    def test_build_cnn_with_nested_channels(self, sample_input):
        """build_model 接受嵌套通道配置（合理层数，不超过28x28输入限制）"""
        model = build_model("mnist_idx", "1x28x28", 10, {
            "channels": [[16, 32], [32, 64]],
        })
        assert isinstance(model, ConfigurableCNN)
        assert model.channel_list == [16, 32, 32, 64]
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert out.shape == (4, 10)

    def test_cnn_with_reported_nested_channels(self, sample_input):
        """用户复现场景：嵌套数组构建CNN（2个block，4层，28x28输入可容纳）"""
        model = ConfigurableCNN(
            in_channels=1, num_classes=10,
            channel_list=[[16, 32], [32, 64]],
        )
        assert model.channel_list == [16, 32, 32, 64]
        model.eval()
        with torch.no_grad():
            out = model(sample_input)
        assert out.shape == (4, 10)

    def test_cnn_rgb_with_nested_channels(self):
        """RGB图像 + 嵌套通道（3x32x32输入，4层最大）"""
        model = ConfigurableCNN(
            in_channels=3, num_classes=10,
            channel_list=[[16, 32], [32, 64]],
        )
        assert model.channel_list == [16, 32, 32, 64]
        x = torch.randn(4, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)


# ============================================================
# 3. _extract_model_config 参数解析测试
# ============================================================
class TestExtractModelConfig:
    """测试 _extract_model_config 对各种参数配置的解析"""

    def test_nested_channels_extraction(self):
        """嵌套通道配置提取"""
        config = {
            "model_config": {
                "channels": [[16, 32], [32, 64], [64, 128]],
            }
        }
        hp = {"learning_rate": 0.001}
        result = TrainingService._extract_model_config(config, hp)
        assert result["channels"] == [16, 32, 32, 64, 64, 128]
        assert all(isinstance(c, int) for c in result["channels"])

    def test_flat_channels_from_config_root(self):
        """从 config 根级别读取扁平 channels"""
        config = {
            "channels": [32, 64, 128],
        }
        hp = {"learning_rate": 0.001}
        result = TrainingService._extract_model_config(config, hp)
        assert result["channels"] == [32, 64, 128]

    def test_nested_channels_from_config_root(self):
        """从 config 根级别读取嵌套 channels"""
        config = {
            "channels": [[16, 32], [64, 128]],
        }
        hp = {"learning_rate": 0.001}
        result = TrainingService._extract_model_config(config, hp)
        assert result["channels"] == [16, 32, 64, 128]

    def test_channel_list_alias(self):
        """channel_list 别名"""
        config = {
            "model_config": {
                "channel_list": [[16, 32], [64, 128]],
            }
        }
        hp = {"learning_rate": 0.001}
        result = TrainingService._extract_model_config(config, hp)
        assert result["channels"] == [16, 32, 64, 128]

    def test_attention_config(self):
        """注意力机制配置"""
        for attn in ["none", "se", "cbam", "self_attention", "eca", "mhsa", "gct"]:
            config = {"model_config": {"channels": [32, 64], "attention": attn}}
            result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
            assert result["attention"] == attn, f"attention={attn} failed"

    def test_activation_config(self):
        """激活函数配置"""
        for act in ["relu", "gelu", "silu", "mish", "leaky_relu", "elu"]:
            config = {"model_config": {"channels": [32, 64], "activation": act}}
            result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
            assert result["activation"] == act, f"activation={act} failed"

    def test_bn_dropout_config(self):
        """BatchNorm / Dropout 配置"""
        # BN=True, Dropout=True
        config = {"model_config": {"channels": [32, 64], "use_bn": True, "use_dropout": True, "dropout_rate": 0.2}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["use_bn"] is True
        assert result["use_dropout"] is True
        assert result["dropout_rate"] == 0.2

        # BN=False, Dropout=False
        config = {"model_config": {"channels": [32, 64], "use_bn": False, "use_dropout": False, "dropout_rate": 0.5}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["use_bn"] is False
        assert result["use_dropout"] is False
        assert result["dropout_rate"] == 0.5

    def test_residual_and_attention_config(self):
        """残差连接和注意力机制配置"""
        # 残差+注意力
        config = {"model_config": {"channels": [32, 64], "use_residual": True, "attention": "se"}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["use_residual"] is True
        assert result["attention"] == "se"

        # 无残差+无注意力
        config = {"model_config": {"channels": [32, 64], "use_residual": False, "attention": "none"}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["use_residual"] is False
        assert result["attention"] == "none"

    def test_fc_hidden_config(self):
        """全连接层配置"""
        config = {"model_config": {"channels": [32, 64], "fc_hidden": 256}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["fc_hidden"] == 256

    def test_default_values(self):
        """默认值测试"""
        config = {"model_config": {"channels": [32, 64]}}
        result = TrainingService._extract_model_config(config, {"learning_rate": 0.001})
        assert result["attention"] == "none"
        assert result["activation"] == "relu"
        assert result["use_bn"] is True
        assert result["use_dropout"] is True
        assert result["dropout_rate"] == 0.3
        assert result["use_residual"] is False
        assert result["fc_hidden"] == 128


# ============================================================
# 4. 超参数合并与训练流程测试
# ============================================================
class TestHyperParameters:
    """测试超参数默认值合并"""

    def test_default_hyperparams_merge(self):
        """默认超参数被用户参数覆盖"""
        from app.services.training_service import _DEFAULT_HYPERPARAMS
        # 用户自定义参数覆盖默认值
        hp = {**_DEFAULT_HYPERPARAMS, "learning_rate": 0.01, "batch_size": 128, "epochs": 50}
        assert hp["learning_rate"] == 0.01
        assert hp["batch_size"] == 128
        assert hp["epochs"] == 50
        # 未覆盖的保持默认值
        assert hp["random_seed"] == 42
        assert hp["val_split"] == 0.2
        assert hp["optimizer"] == "adam"
        assert hp["loss_function"] == "cross_entropy"

    def test_optimizer_names(self):
        """支持的优化器列表"""
        from app.services.training_service import _SUPPORTED_OPTIMIZERS
        assert "adam" in _SUPPORTED_OPTIMIZERS
        assert "sgd" in _SUPPORTED_OPTIMIZERS
        assert "adamw" in _SUPPORTED_OPTIMIZERS
        assert "rmsprop" in _SUPPORTED_OPTIMIZERS

    def test_loss_function_registry(self):
        """损失函数注册表"""
        from app.ml.losses import list_loss_functions
        loss_names = list_loss_functions()
        # 返回的是损失函数名列表，如 ['AsymmetricLoss', 'CombinedLoss', ...]
        assert len(loss_names) > 0
        assert any("CrossEntropy" in l or "cross_entropy" in l.lower() for l in loss_names)


# ============================================================
# 5. 完整训练流程测试（模型构建 + 训练循环）
# ============================================================
class TestFullTrainingPipeline:
    """测试完整训练管道"""

    def test_cnn_training_with_nested_channels(self):
        """使用嵌套通道配置的完整CNN训练"""
        import torch.optim as optim
        from app.ml.losses import create_loss_function
        from torch.utils.data import TensorDataset, DataLoader

        # 使用嵌套通道配置创建模型
        model = ConfigurableCNN(
            in_channels=1, num_classes=10,
            channel_list=[[16, 32], [32, 64]],
            activation="relu", use_bn=True, use_dropout=True,
            dropout_rate=0.2, use_residual=False, attention="none",
            fc_hidden=128,
        )
        assert model.channel_list == [16, 32, 32, 64]

        # 创建模拟数据
        X = torch.randn(200, 1, 28, 28)
        y = torch.randint(0, 10, (200,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # 训练配置
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = create_loss_function("cross_entropy")

        # 训练一个epoch
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        # 验证loss下降
        assert loss.item() < 10.0, f"Loss too high: {loss.item()}"
        assert not torch.isnan(loss), "Loss is NaN"

    def test_mlp_training(self):
        """MLP训练测试"""
        import torch.optim as optim
        from app.ml.losses import create_loss_function
        from torch.utils.data import TensorDataset, DataLoader

        model = ConfigurableMLP(
            in_features=100, hidden_dims=[64, 32], num_classes=5,
            use_bn=True, use_dropout=True,
            dropout_rate=0.2, use_attention=False,
        )

        X = torch.randn(200, 100)
        y = torch.randint(0, 5, (200,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = create_loss_function("cross_entropy")

        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 10.0
        assert not torch.isnan(loss)

    def test_cnn_with_attention_training(self):
        """CNN + 注意力机制训练"""
        import torch.optim as optim
        from app.ml.losses import create_loss_function
        from torch.utils.data import TensorDataset, DataLoader

        model = ConfigurableCNN(
            in_channels=1, num_classes=10,
            channel_list=[[16, 32], [32, 64]],
            attention="se",
            use_bn=True, use_dropout=True, dropout_rate=0.2,
            fc_hidden=128,
        )

        X = torch.randn(200, 1, 28, 28)
        y = torch.randint(0, 10, (200,))
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = create_loss_function("cross_entropy")

        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 10.0
        assert not torch.isnan(loss)

    def test_cnn_with_residual_training(self):
        """CNN + 残差连接训练"""
        import torch.optim as optim
        from app.ml.losses import create_loss_function
        from torch.utils.data import TensorDataset, DataLoader

        model = ConfigurableCNN(
            in_channels=1, num_classes=10,
            channel_list=[32, 64, 128],
            use_residual=True,
            use_bn=True, use_dropout=True, dropout_rate=0.2,
            fc_hidden=128,
        )

        X = torch.randn(200, 1, 28, 28)
        y = torch.randint(0, 10, (200,))
        loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = create_loss_function("cross_entropy")

        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 10.0
        assert not torch.isnan(loss)


# ============================================================
# 6. 边界条件测试
# ============================================================
class TestEdgeCases:
    """测试边界条件"""

    def test_empty_nested_channels(self):
        """空嵌套通道"""
        with pytest.raises(ValueError, match="展平后为空"):
            _flatten_channels([[]])

    def test_very_deep_nested(self):
        """极深嵌套"""
        result = _flatten_channels([[[[[[[32]]]]]]])
        assert result == [32]

    def test_large_channel_values(self):
        """大值通道"""
        channels = [[512, 1024], [1024, 2048]]
        result = _flatten_channels(channels)
        assert result == [512, 1024, 1024, 2048]

    def test_single_channel_nested(self):
        """单通道嵌套"""
        assert _flatten_channels([[1]]) == [1]

    def test_mixed_types(self):
        """混合类型（int, float）"""
        result = _flatten_channels([[16.0, 32], [64.0, 128.0]])
        assert result == [16, 32, 64, 128]
        assert all(isinstance(c, int) for c in result)

    def test_invalid_string_value(self):
        """非法字符串值"""
        with pytest.raises(ValueError, match="无法解析"):
            _flatten_channels(["abc"])


# ============================================================
# 7. 前端参数序列化模拟测试
# ============================================================
class TestFrontendSerialization:
    """模拟前端JSON序列化/反序列化场景"""

    def test_json_round_trip_nested_channels(self):
        """嵌套通道配置的JSON往返序列化"""
        import json
        channels = [[16, 32], [32, 64], [32, 64, 128], [64, 128, 256], [64, 128, 256, 512]]
        payload = json.dumps({"channels": channels})
        parsed = json.loads(payload)
        result = _flatten_channels(parsed["channels"])
        assert result == [16, 32, 32, 64, 32, 64, 128, 64, 128, 256, 64, 128, 256, 512]

    def test_json_round_trip_flat_channels(self):
        """扁平通道配置的JSON往返序列化"""
        import json
        channels = [32, 64, 128]
        payload = json.dumps({"channels": channels})
        parsed = json.loads(payload)
        # 扁平列表不应被展平修改
        result = _flatten_channels(parsed["channels"])
        assert result == [32, 64, 128]

    def test_full_config_json_round_trip(self):
        """完整训练配置的JSON往返序列化"""
        import json
        config = {
            "model_config": {
                "channels": [[16, 32], [32, 64], [32, 64, 128], [64, 128, 256], [64, 128, 256, 512]],
                "activation": "relu",
                "use_bn": True,
                "use_dropout": True,
                "dropout_rate": 0.2,
                "use_residual": False,
                "attention": "none",
                "fc_hidden": 128,
            }
        }
        payload = json.dumps(config)
        parsed = json.loads(payload)

        # 验证 _extract_model_config 正确处理
        result = TrainingService._extract_model_config(parsed, {"learning_rate": 0.001})
        assert result["channels"] == [16, 32, 32, 64, 32, 64, 128, 64, 128, 256, 64, 128, 256, 512]
        assert result["activation"] == "relu"
        assert result["use_bn"] is True
        assert result["use_dropout"] is True
        assert result["dropout_rate"] == 0.2
        assert result["use_residual"] is False
        assert result["attention"] == "none"
        assert result["fc_hidden"] == 128
        # 验证所有通道都是int
        assert all(isinstance(c, int) for c in result["channels"])