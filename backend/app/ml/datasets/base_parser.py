"""
数据集解析器抽象基类
所有格式解析器继承该基类，实现统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class DatasetParseError(Exception):
    """数据集解析异常"""

    def __init__(self, format_name: str, message: str):
        self.format_name = format_name
        self.message = message
        super().__init__(f"[{format_name}] 解析失败: {message}")


class UnrecognizedFormatError(Exception):
    """未识别数据集格式异常"""

    def __init__(self):
        super().__init__(
            "未识别到有效数据集格式，支持：图片目录集、MNIST IDX二进制、NumPy数组、CSV表格"
        )


class BaseDatasetParser(ABC):
    """数据集解析器抽象基类"""

    # 格式标识，子类必须覆盖
    dataset_type: str = "base"

    @staticmethod
    @abstractmethod
    def detect_format(extract_path: str) -> bool:
        """检测解压目录是否匹配当前格式

        Args:
            extract_path: 解压后的目录路径

        Returns:
            bool: 是否匹配该格式
        """
        ...

    @abstractmethod
    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析元数据，返回统一结构

        Returns:
            dict with keys:
                - sample_count: 总样本数
                - class_count: 类别总数
                - feature_shape: 特征维度/图像尺寸（字符串，如 "28x28x1"、"784"）
                - class_distribution: dict，每个类别的样本数
                - dataset_type: 格式标识
        """
        ...

    @abstractmethod
    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载数据，返回 (X, y) numpy 数组

        Args:
            extract_path: 解压后的目录路径
            split: 数据集分片，'train' / 'test' / 'val'

        Returns:
            (X, y): 特征数组和标签数组
        """
        ...
