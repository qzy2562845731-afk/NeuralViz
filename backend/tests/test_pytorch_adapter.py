"""
PyTorch 模型适配器单元测试
覆盖: 模型加载、层解析、推理、激活值获取
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np
from app.ml.pytorch_adapter import PyTorchAdapter


class SimpleTestModel(nn.Module):
    """用于测试的简单 CNN 模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestPyTorchAdapter:
    """PyTorch 适配器测试"""

    @pytest.fixture
    def adapter(self):
        return PyTorchAdapter()

    @pytest.fixture
    def temp_model_path(self):
        """创建临时模型文件"""
        model = SimpleTestModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
            yield f.name
        os.unlink(f.name)

    def test_load_model_pt(self, adapter, temp_model_path):
        """加载 .pt 模型"""
        result = adapter.load_model(temp_model_path)
        assert result is True
        assert adapter.model is not None
        assert adapter.model_path == temp_model_path

    def test_load_model_state_dict(self, adapter):
        """加载 state_dict（需要模型结构）"""
        model = SimpleTestModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"state_dict": model.state_dict()}, f.name)
        try:
            # state_dict 无模型结构，应该抛出异常
            with pytest.raises(Exception):
                adapter.load_model(f.name)
        finally:
            os.unlink(f.name)

    def test_load_model_with_model_key(self, adapter):
        """加载带 model 键的 dict"""
        model = SimpleTestModel()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": model}, f.name)
        try:
            result = adapter.load_model(f.name)
            assert result is True
            assert adapter.model is not None
        finally:
            os.unlink(f.name)

    def test_get_layer_info(self, adapter, temp_model_path):
        """获取层信息"""
        adapter.load_model(temp_model_path)
        layers = adapter.get_layer_info()
        assert len(layers) > 0
        # 应该包含 Input 层
        assert layers[0]["type"] == "Input"
        # 应该包含 Conv2d 层
        conv_layers = [l for l in layers if l["type"] == "Conv2d"]
        assert len(conv_layers) >= 2

    def test_get_input_shape(self, adapter, temp_model_path):
        """获取输入形状"""
        adapter.load_model(temp_model_path)
        shape = adapter.get_input_shape()
        # 模型输入通道数为 1，使用 28x28 默认
        assert shape == (1, 1, 28, 28)

    def test_get_output_shape(self, adapter, temp_model_path):
        """获取输出形状"""
        adapter.load_model(temp_model_path)
        shape = adapter.get_output_shape()
        assert len(shape) == 2
        assert shape[1] == 5  # 5 类

    def test_infer(self, adapter, temp_model_path):
        """执行推理"""
        adapter.load_model(temp_model_path)
        input_data = np.random.rand(1, 28, 28).astype(np.float32)
        output, inference_time = adapter.infer(input_data)
        assert output.shape == (1, 5)
        assert inference_time >= 0

    def test_infer_with_batch(self, adapter, temp_model_path):
        """批量推理"""
        adapter.load_model(temp_model_path)
        input_data = np.random.rand(4, 1, 28, 28).astype(np.float32)
        output, _ = adapter.infer(input_data)
        assert output.shape == (4, 5)

    def test_get_activations(self, adapter, temp_model_path):
        """获取层激活值"""
        adapter.load_model(temp_model_path)
        input_data = np.random.rand(1, 28, 28).astype(np.float32)
        act = adapter.get_activations("conv1", input_data)
        assert act is not None
        assert act.shape[0] == 1  # batch size

    def test_get_all_activations(self, adapter, temp_model_path):
        """获取所有层激活值"""
        adapter.load_model(temp_model_path)
        input_data = np.random.rand(1, 28, 28).astype(np.float32)
        acts = adapter.get_all_activations(input_data)
        assert len(acts) > 0

    def test_infer_without_model(self, adapter):
        """未加载模型时推理"""
        with pytest.raises(RuntimeError):
            adapter.infer(np.random.rand(1, 28, 28).astype(np.float32))

    def test_layer_info_params(self, adapter, temp_model_path):
        """层信息包含参数量"""
        adapter.load_model(temp_model_path)
        layers = adapter.get_layer_info()
        conv_layers = [l for l in layers if l["type"] == "Conv2d"]
        for cl in conv_layers:
            assert cl["params"] > 0

    def test_layer_info_conv_details(self, adapter, temp_model_path):
        """Conv2d 层详细信息"""
        adapter.load_model(temp_model_path)
        layers = adapter.get_layer_info()
        conv1 = next(l for l in layers if l["name"] == "conv1")
        assert conv1["filters"] == 8
        assert "kernel_size" in conv1

    def test_layer_info_linear_details(self, adapter, temp_model_path):
        """Linear 层详细信息"""
        adapter.load_model(temp_model_path)
        layers = adapter.get_layer_info()
        fc = next(l for l in layers if l["name"] == "fc")
        assert fc["out_features"] == 5
        assert "in_features" in fc

    def test_rgb_model_input_shape(self, adapter):
        """RGB 模型输入形状（3 通道 <= 4 使用 28x28）"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
        try:
            adapter.load_model(f.name)
            shape = adapter.get_input_shape()
            # 3 <= 4 通道，使用 28x28
            assert shape == (1, 3, 28, 28)
        finally:
            os.unlink(f.name)

    def test_large_channel_model_input_shape(self, adapter):
        """大通道数模型输入形状（> 4 通道使用 224x224）"""
        model = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
        try:
            adapter.load_model(f.name)
            shape = adapter.get_input_shape()
            # 64 > 4 通道，使用 224x224
            assert shape == (1, 64, 224, 224)
        finally:
            os.unlink(f.name)