"""
数据集解析器工厂
按优先级自动识别格式并返回对应解析器实例
"""
from typing import List, Type, Optional
from .base_parser import BaseDatasetParser, UnrecognizedFormatError
from .mnist_parser import MnistIDXParser
from .numpy_parser import NumpyParser
from .csv_parser import CSVParser
from .json_parser import JSONParser
from .image_folder_parser import ImageFolderParser


# 解析器注册表（按识别优先级排序）
_PARSER_REGISTRY: List[Type[BaseDatasetParser]] = [
    MnistIDXParser,     # 优先级 1：MNIST IDX 二进制
    NumpyParser,        # 优先级 2：NumPy 数组
    CSVParser,          # 优先级 3：CSV/TSV 表格
    JSONParser,         # 优先级 4：JSON 格式
    ImageFolderParser,  # 优先级 5：图片目录（兜底）
]


class DatasetParserFactory:
    """数据集解析器工厂"""

    @staticmethod
    def detect_and_create(extract_path: str) -> BaseDatasetParser:
        """自动识别数据集格式并返回对应解析器实例

        按优先级依次调用各解析器的 detect_format 方法，
        匹配到第一个符合的格式后返回对应解析器实例。

        Args:
            extract_path: 解压后的目录路径

        Returns:
            BaseDatasetParser: 匹配到的解析器实例

        Raises:
            UnrecognizedFormatError: 所有格式均不匹配时抛出
        """
        for parser_cls in _PARSER_REGISTRY:
            try:
                if parser_cls.detect_format(extract_path):
                    return parser_cls()
            except Exception:
                # detect_format 异常时跳过该解析器，尝试下一个
                continue

        raise UnrecognizedFormatError()

    @staticmethod
    def get_supported_formats() -> List[str]:
        """获取所有支持的格式标识"""
        return [p.dataset_type for p in _PARSER_REGISTRY]

    @staticmethod
    def register_parser(parser_cls: Type[BaseDatasetParser], priority: int = None):
        """注册新解析器（开闭原则：后续新增格式只需调用此方法注册）

        Args:
            parser_cls: 解析器类
            priority: 插入位置（优先级），None 表示追加到末尾
        """
        if priority is not None:
            _PARSER_REGISTRY.insert(priority, parser_cls)
        else:
            _PARSER_REGISTRY.append(parser_cls)
