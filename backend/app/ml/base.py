from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class BaseModelAdapter(ABC):
    """
    模型适配器抽象基类
    定义统一的模型加载、解析、推理接口
    """
    
    def __init__(self):
        self.model = None
        self.model_path: Optional[str] = None
        self.model_name: Optional[str] = None
        self._layer_info: List[Dict] = []
        # 推理输出是否已是概率分布（已含 softmax/sigmoid）。
        # True  -> 推理 API 跳过 softmax 后处理
        # False -> 推理 API 对 logits 应用 softmax
        # 各子类在 load_model 后应据此属性声明输出约定，替代脆弱的启发式判断。
        self.output_is_probability: bool = False
        # 模型输入布局："NCHW"（PyTorch 默认）或 "NHWC"（TF/部分 ONNX 模型）。
        # ONNXAdapter 加载时自动检测；其他适配器按需覆盖。
        self.input_layout: str = "NCHW"
    
    @abstractmethod
    def load_model(self, file_path: str) -> bool:
        """
        加载模型文件
        
        Args:
            file_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        pass
    
    @abstractmethod
    def get_layer_info(self) -> List[Dict[str, Any]]:
        """
        获取所有层的详细信息
        
        Returns:
            层信息列表，每项包含：
            - id: 层索引
            - name: 层名称
            - type: 层类型 (Conv2d, Linear, MaxPool, etc.)
            - input_shape: 输入形状
            - output_shape: 输出形状
            - params: 参数量
            - kernel_size: 卷积核尺寸（如果有）
            - filters: 卷积核数量（如果有）
            - stride: 步长（如果有）
            - padding: 填充（如果有）
        """
        pass
    
    @abstractmethod
    def infer(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        执行前向推理
        
        Args:
            input_data: 输入数据 (H, W, C) 或 (N, C, H, W)
            
        Returns:
            (预测结果, 推理耗时(秒))
        """
        pass
    
    @abstractmethod
    def get_activations(self, layer_name: str, input_data: np.ndarray) -> np.ndarray:
        """
        获取指定层的激活值
        
        Args:
            layer_name: 层名称
            input_data: 输入数据
            
        Returns:
            激活值数组
        """
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """获取模型输入形状"""
        pass
    
    @abstractmethod
    def get_output_shape(self) -> Tuple[int, ...]:
        """获取模型输出形状"""
        pass
    
    @property
    def total_params(self) -> int:
        """获取模型总参数量"""
        return sum(layer.get("params", 0) for layer in self._layer_info)
    
    @property
    def layer_count(self) -> int:
        """获取模型层数"""
        return len(self._layer_info)