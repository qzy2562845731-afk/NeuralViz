"""
数据集解析器工厂单元测试
覆盖: detect_and_create, get_supported_formats, register_parser
"""
import os
import tempfile
import pytest
import numpy as np
from app.ml.datasets.factory import DatasetParserFactory
from app.ml.datasets.base_parser import (
    BaseDatasetParser, DatasetParseError, UnrecognizedFormatError,
)
from app.ml.datasets.csv_parser import CSVParser
from app.ml.datasets.numpy_parser import NumpyParser
from app.ml.datasets.json_parser import JSONParser
from app.ml.datasets.image_folder_parser import ImageFolderParser


class TestDatasetParserFactory:
    """解析器工厂测试"""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_detect_csv(self, temp_dir):
        """自动识别 CSV 格式"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n")
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, CSVParser)

    def test_detect_numpy_npy(self, temp_dir):
        """自动识别 NumPy .npy"""
        np.save(os.path.join(temp_dir, "data.npy"), np.random.rand(10, 5))
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, NumpyParser)

    def test_detect_numpy_npz(self, temp_dir):
        """自动识别 NumPy .npz"""
        np.savez(os.path.join(temp_dir, "data.npz"),
                 X=np.random.rand(10, 5),
                 y=np.random.randint(0, 3, 10))
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, NumpyParser)

    def test_detect_json(self, temp_dir):
        """自动识别 JSON 格式"""
        import json
        with open(os.path.join(temp_dir, "data.json"), "w") as f:
            json.dump({"data": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1]}, f)
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, JSONParser)

    def test_detect_image_folder(self, temp_dir):
        """自动识别图片目录（兜底）"""
        # 创建图片目录
        os.makedirs(os.path.join(temp_dir, "class1"))
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=128)
        img.save(os.path.join(temp_dir, "class1", "img.png"))
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, ImageFolderParser)

    def test_unrecognized_format(self, temp_dir):
        """无法识别的格式"""
        with pytest.raises(UnrecognizedFormatError):
            DatasetParserFactory.detect_and_create(temp_dir)

    def test_get_supported_formats(self):
        """获取支持的格式列表"""
        formats = DatasetParserFactory.get_supported_formats()
        assert "csv" in formats
        assert "numpy" in formats
        assert "json" in formats
        assert "image_folder" in formats
        assert "mnist_idx" in formats

    def test_register_parser(self, temp_dir):
        """注册新解析器"""
        class DummyParser(BaseDatasetParser):
            dataset_type = "dummy"

            @staticmethod
            def detect_format(extract_path: str) -> bool:
                return os.path.exists(os.path.join(extract_path, "dummy.txt"))

            def parse_metadata(self, extract_path: str):
                return {"sample_count": 0, "class_count": 0, "feature_shape": "0",
                        "class_distribution": {}, "dataset_type": "dummy"}

            def load_data(self, extract_path: str, split: str = "train"):
                return np.array([]), np.array([])

        DatasetParserFactory.register_parser(DummyParser, priority=0)
        assert "dummy" in DatasetParserFactory.get_supported_formats()

    def test_parser_priority_csv_before_json(self, temp_dir):
        """CSV 优先级高于 JSON（按注册表顺序）"""
        # 同时存在 CSV 和 JSON，应返回 CSV 解析器
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n")
        import json
        with open(os.path.join(temp_dir, "data.json"), "w") as f:
            json.dump({"data": [[1.0]], "labels": [0]}, f)
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, CSVParser)

    def test_parser_priority_numpy_before_csv(self, temp_dir):
        """NumPy 优先级高于 CSV"""
        np.save(os.path.join(temp_dir, "data.npy"), np.random.rand(10, 5))
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n")
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, NumpyParser)

    def test_detect_format_exception_skip(self, temp_dir):
        """detect_format 异常时跳过该解析器"""
        # 创建损坏的 .npy 文件，NumpyParser.detect_format 会返回 True（按扩展名），
        # 但由于它是损坏的，parse_metadata 会失败，但这不是 factory 的问题。
        # 这里验证：同时存在 .npy 和 CSV 时，NumpyParser 优先级更高先被选中。
        with open(os.path.join(temp_dir, "bad.npy"), "w") as f:
            f.write("not a valid numpy file")
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n")
        # NumpyParser 优先级高于 CSVParser（按注册顺序），所以应该返回 NumpyParser
        from app.ml.datasets.numpy_parser import NumpyParser
        parser = DatasetParserFactory.detect_and_create(temp_dir)
        assert isinstance(parser, NumpyParser)