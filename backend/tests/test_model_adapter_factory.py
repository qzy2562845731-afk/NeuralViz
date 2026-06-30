"""
模型适配器工厂单元测试
覆盖: create, is_supported, get_supported_formats
"""
import pytest
from app.ml.factory import ModelAdapterFactory


class TestModelAdapterFactory:
    """模型适配器工厂测试"""

    def test_is_supported_pt(self):
        """支持 .pt 格式"""
        assert ModelAdapterFactory.is_supported("model.pt") is True

    def test_is_supported_pth(self):
        """支持 .pth 格式"""
        assert ModelAdapterFactory.is_supported("model.pth") is True

    def test_is_supported_onnx(self):
        """支持 .onnx 格式"""
        assert ModelAdapterFactory.is_supported("model.onnx") is True

    def test_is_supported_h5(self):
        """支持 .h5 格式"""
        assert ModelAdapterFactory.is_supported("model.h5") is True

    def test_is_supported_pkl(self):
        """支持 .pkl 格式"""
        assert ModelAdapterFactory.is_supported("model.pkl") is True

    def test_is_supported_pickle(self):
        """支持 .pickle 格式"""
        assert ModelAdapterFactory.is_supported("model.pickle") is True

    def test_is_supported_unsupported(self):
        """不支持的格式"""
        assert ModelAdapterFactory.is_supported("model.txt") is False
        assert ModelAdapterFactory.is_supported("model.bin") is False

    def test_is_supported_builtin(self):
        """内置示例模型"""
        assert ModelAdapterFactory.is_supported("sample_cnn") is True
        assert ModelAdapterFactory.is_supported("sample") is True
        assert ModelAdapterFactory.is_supported("default") is True

    def test_is_supported_case_insensitive(self):
        """大小写不敏感"""
        assert ModelAdapterFactory.is_supported("model.PT") is True
        assert ModelAdapterFactory.is_supported("model.ONNX") is True

    def test_get_supported_formats(self):
        """获取支持的格式列表"""
        formats = ModelAdapterFactory.get_supported_formats()
        assert len(formats) > 0
        ext_list = [f["ext"] for f in formats]
        assert ".pt" in ext_list
        assert ".pth" in ext_list
        assert ".onnx" in ext_list

    def test_create_pt_adapter(self):
        """创建 PyTorch 适配器"""
        from app.ml.pytorch_adapter import PyTorchAdapter
        import tempfile
        import torch
        import torch.nn as nn
        import os

        model = nn.Sequential(nn.Linear(10, 5))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model, f.name)
        try:
            adapter = ModelAdapterFactory.create(f.name)
            assert isinstance(adapter, PyTorchAdapter)
        finally:
            os.unlink(f.name)

    def test_create_unsupported_format(self):
        """不支持的格式抛异常"""
        from app.core.exception import UnsupportedFormatException
        with pytest.raises(UnsupportedFormatException):
            ModelAdapterFactory.create("model.txt")

    def test_create_pth_adapter(self):
        """创建 .pth 适配器"""
        from app.ml.pytorch_adapter import PyTorchAdapter
        import tempfile
        import torch
        import torch.nn as nn
        import os

        model = nn.Sequential(nn.Linear(10, 5))
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model, f.name)
        try:
            adapter = ModelAdapterFactory.create(f.name)
            assert isinstance(adapter, PyTorchAdapter)
        finally:
            os.unlink(f.name)