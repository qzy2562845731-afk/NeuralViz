"""
NumPy 数组解析器单元测试
覆盖: detect_format, parse_metadata, load_data, .npy / .npz 格式
"""
import os
import tempfile
import pytest
import numpy as np
from app.ml.datasets.numpy_parser import NumpyParser
from app.ml.datasets.base_parser import DatasetParseError


class TestNumpyParser:
    """NumPy 解析器测试"""

    @pytest.fixture
    def parser(self):
        return NumpyParser()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_dataset_type(self, parser):
        assert parser.dataset_type == "numpy"

    def test_detect_format_npy(self, parser, temp_dir):
        """检测 .npy 文件"""
        np.save(os.path.join(temp_dir, "data.npy"), np.random.rand(10, 5))
        assert NumpyParser.detect_format(temp_dir) is True

    def test_detect_format_npz(self, parser, temp_dir):
        """检测 .npz 文件"""
        np.savez(os.path.join(temp_dir, "data.npz"), X=np.random.rand(10, 5), y=np.random.randint(0, 3, 10))
        assert NumpyParser.detect_format(temp_dir) is True

    def test_detect_format_negative(self, parser, temp_dir):
        """无 NumPy 文件"""
        assert NumpyParser.detect_format(temp_dir) is False

    def test_parse_metadata_npy(self, parser, temp_dir):
        """解析 .npy 元数据"""
        X = np.random.rand(100, 20).astype(np.float32)
        y = np.random.randint(0, 5, 100).astype(np.int64)
        data = np.column_stack([X, y])
        np.save(os.path.join(temp_dir, "data.npy"), data)
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 100
        assert stats["feature_shape"] == "20"
        assert stats["dataset_type"] == "numpy"

    def test_parse_metadata_npz(self, parser, temp_dir):
        """解析 .npz 元数据"""
        X = np.random.rand(50, 10).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), X=X, y=y)
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 50
        assert stats["feature_shape"] == "10"
        assert stats["dataset_type"] == "numpy"

    def test_parse_metadata_npz_alt_keys(self, parser, temp_dir):
        """.npz 使用备用键名"""
        X = np.random.rand(30, 8).astype(np.float32)
        y = np.random.randint(0, 2, 30).astype(np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), data=X, labels=y)
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 30
        assert stats["feature_shape"] == "8"

    def test_parse_metadata_npz_features_target(self, parser, temp_dir):
        """.npz 使用 features/target 键名"""
        X = np.random.rand(20, 12).astype(np.float32)
        y = np.random.randint(0, 4, 20).astype(np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), features=X, target=y)
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 20
        assert stats["feature_shape"] == "12"

    def test_parse_metadata_no_files(self, parser, temp_dir):
        """无 NumPy 文件抛异常"""
        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_parse_metadata_1d_npy(self, parser, temp_dir):
        """一维 .npy 数组"""
        data = np.random.rand(100).astype(np.float32)
        np.save(os.path.join(temp_dir, "data.npy"), data)
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 100

    def test_load_data_npy(self, parser, temp_dir):
        """加载 .npy 数据"""
        X = np.random.rand(50, 20).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int64)
        data = np.column_stack([X, y])
        np.save(os.path.join(temp_dir, "data.npy"), data)
        X_out, y_out = parser.load_data(temp_dir)
        assert X_out.shape == (50, 20)
        assert y_out.shape == (50,)
        assert np.array_equal(y_out, y)

    def test_load_data_npz(self, parser, temp_dir):
        """加载 .npz 数据"""
        X = np.random.rand(40, 15).astype(np.float32)
        y = np.random.randint(0, 5, 40).astype(np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), X=X, y=y)
        X_out, y_out = parser.load_data(temp_dir)
        assert X_out.shape == (40, 15)
        assert y_out.shape == (40,)

    def test_load_data_npz_single_array(self, parser, temp_dir):
        """.npz 单数组无标签"""
        X = np.random.rand(25, 10).astype(np.float32)
        np.savez(os.path.join(temp_dir, "data.npz"), X=X)
        X_out, y_out = parser.load_data(temp_dir)
        assert X_out.shape == (25, 10)
        assert y_out.shape == (25,)

    def test_load_data_multiple_npy(self, parser, temp_dir):
        """多个 .npy 文件按名称匹配"""
        X = np.random.rand(30, 8).astype(np.float32)
        y = np.random.randint(0, 2, 30).astype(np.int64)
        np.save(os.path.join(temp_dir, "X_data.npy"), X)
        np.save(os.path.join(temp_dir, "y_labels.npy"), y)
        X_out, y_out = parser.load_data(temp_dir)
        assert X_out.shape == (30, 8)
        assert y_out.shape == (30,)

    def test_load_data_split_selection(self, parser, temp_dir):
        """多 .npz split 选择"""
        np.savez(os.path.join(temp_dir, "train.npz"),
                 X=np.random.rand(50, 10).astype(np.float32),
                 y=np.random.randint(0, 3, 50))
        np.savez(os.path.join(temp_dir, "test.npz"),
                 X=np.random.rand(20, 10).astype(np.float32),
                 y=np.random.randint(0, 3, 20))
        X_train, _ = parser.load_data(temp_dir, split="train")
        X_test, _ = parser.load_data(temp_dir, split="test")
        assert X_train.shape == (50, 10)
        assert X_test.shape == (20, 10)

    def test_class_distribution(self, parser, temp_dir):
        """类别分布统计"""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.array([0] * 60 + [1] * 40, dtype=np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), X=X, y=y)
        stats = parser.parse_metadata(temp_dir)
        dist = stats["class_distribution"]
        assert dist["0"] == 60
        assert dist["1"] == 40

    def test_feature_shape_image(self, parser, temp_dir):
        """图像数据 feature_shape"""
        X = np.random.rand(100, 1, 28, 28).astype(np.float32)
        y = np.random.randint(0, 10, 100).astype(np.int64)
        np.savez(os.path.join(temp_dir, "data.npz"), X=X, y=y)
        stats = parser.parse_metadata(temp_dir)
        assert stats["feature_shape"] == "1x28x28"