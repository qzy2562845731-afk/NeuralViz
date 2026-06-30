from pathlib import Path
from typing import Optional, Dict, Type

from .base import BaseModelAdapter
from app.core.exception import UnsupportedFormatException


class ModelAdapterFactory:
    """
    模型适配器工厂
    根据文件后缀自动匹配对应适配器
    采用延迟导入策略，避免启动时加载所有 ML 依赖
    """

    # 后缀到适配器类路径的映射（延迟导入）
    _ADAPTER_MAP: Dict[str, str] = {
        ".onnx": ".onnx_adapter:ONNXAdapter",
        ".pb": ".onnx_adapter:ONNXAdapter",
        ".pt": ".pytorch_adapter:PyTorchAdapter",
        ".pth": ".pytorch_adapter:PyTorchAdapter",
        ".h5": ".pytorch_adapter:PyTorchAdapter",
        ".keras": ".pytorch_adapter:PyTorchAdapter",
        ".hdf5": ".pytorch_adapter:PyTorchAdapter",
        ".pickle": ".pytorch_adapter:PyTorchAdapter",
        ".pkl": ".pytorch_adapter:PyTorchAdapter",
    }

    # 内置示例模型（虚拟文件路径到适配器的映射）
    _BUILTIN_MODELS: Dict[str, str] = {
        "sample_cnn": ".sample_cnn_adapter:SampleCNNAdapter",
        "sample": ".sample_cnn_adapter:SampleCNNAdapter",
        "default": ".sample_cnn_adapter:SampleCNNAdapter",
    }

    # 缓存已加载的适配器类
    _adapter_cache: Dict[str, Type[BaseModelAdapter]] = {}

    @classmethod
    def _load_adapter(cls, adapter_path: str) -> Type[BaseModelAdapter]:
        """延迟加载适配器类"""
        if adapter_path in cls._adapter_cache:
            return cls._adapter_cache[adapter_path]

        module_path, class_name = adapter_path.split(":")

        try:
            # 动态导入
            import importlib
            module = importlib.import_module(module_path, package=__package__)
            adapter_class = getattr(module, class_name)
            cls._adapter_cache[adapter_path] = adapter_class
            return adapter_class
        except ImportError as e:
            raise UnsupportedFormatException(
                f"适配器 {class_name} 所需依赖未安装: {str(e)}"
            ) from e

    @classmethod
    def create(cls, file_path: str) -> BaseModelAdapter:
        """
        根据文件路径创建对应的适配器

        支持：
        1. 真实模型文件（按后缀匹配适配器）
        2. 内置示例模型（按名称匹配）
        """
        # 先检查是否是内置示例模型
        model_key = Path(file_path).stem.lower()
        if model_key in cls._BUILTIN_MODELS:
            adapter_class = cls._load_adapter(cls._BUILTIN_MODELS[model_key])
            adapter = adapter_class()
            adapter.load_model(file_path)
            return adapter

        # 按后缀匹配
        suffix = Path(file_path).suffix.lower()

        if suffix not in cls._ADAPTER_MAP:
            supported = ", ".join(cls._ADAPTER_MAP.keys())
            raise UnsupportedFormatException(
                f"不支持的格式: {suffix}，支持的格式: {supported}"
            )

        adapter_class = cls._load_adapter(cls._ADAPTER_MAP[suffix])
        return adapter_class()

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """检查文件格式是否支持（注册层面的支持，不检查依赖是否安装）"""
        model_key = Path(file_path).stem.lower()
        if model_key in cls._BUILTIN_MODELS:
            return True
        suffix = Path(file_path).suffix.lower()
        return suffix in cls._ADAPTER_MAP

    @classmethod
    def get_supported_formats(cls) -> list:
        """获取所有注册的格式列表"""
        return [
            {"ext": ext, "supported": True}
            for ext in cls._ADAPTER_MAP.keys()
        ]
