"""
CSV/TSV 解析器单元测试
覆盖: detect_format, parse_metadata, load_data, _read_csv, _detect_header
"""
import os
import tempfile
import pytest
import numpy as np
from app.ml.datasets.csv_parser import CSVParser
from app.ml.datasets.base_parser import DatasetParseError


class TestCSVParser:
    """CSV 解析器测试"""

    @pytest.fixture
    def parser(self):
        return CSVParser()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_dataset_type(self, parser):
        assert parser.dataset_type == "csv"

    def test_detect_format_positive(self, parser, temp_dir):
        """检测 CSV 文件"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n3.0,4.0,1\n")
        assert CSVParser.detect_format(temp_dir) is True

    def test_detect_format_tsv(self, parser, temp_dir):
        """检测 TSV 文件"""
        with open(os.path.join(temp_dir, "data.tsv"), "w") as f:
            f.write("1.0\t2.0\t0\n3.0\t4.0\t1\n")
        assert CSVParser.detect_format(temp_dir) is True

    def test_detect_format_negative(self, parser, temp_dir):
        """无 CSV 文件时返回 False"""
        assert CSVParser.detect_format(temp_dir) is False

    def test_parse_metadata_numeric(self, parser, temp_dir):
        """解析纯数值 CSV 元数据"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 3
        assert stats["feature_shape"] == "2"
        assert stats["dataset_type"] == "csv"
        assert stats["class_count"] == 2

    def test_parse_metadata_with_header(self, parser, temp_dir):
        """带表头的 CSV"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n")
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 2
        assert stats["feature_shape"] == "2"

    def test_parse_metadata_tsv(self, parser, temp_dir):
        """TSV 格式"""
        with open(os.path.join(temp_dir, "data.tsv"), "w") as f:
            f.write("1.0\t2.0\t0\n3.0\t4.0\t1\n")
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 2
        assert stats["dataset_type"] == "csv"

    def test_parse_metadata_empty_file(self, parser, temp_dir):
        """空文件抛异常"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("")
        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_parse_metadata_no_csv(self, parser, temp_dir):
        """无 CSV 文件"""
        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_load_data(self, parser, temp_dir):
        """加载 CSV 数据"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
        X, y = parser.load_data(temp_dir)
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert list(y) == [0, 1, 0]

    def test_load_data_no_labels(self, parser, temp_dir):
        """单列 CSV 无标签"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0\n2.0\n3.0\n")
        X, y = parser.load_data(temp_dir)
        # 单列数据: ndim=2, data[:,:-1] 返回 (3,0)
        assert X.shape in [(3,), (3, 0)]
        assert y is not None

    def test_load_data_with_header(self, parser, temp_dir):
        """带表头加载"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("f1,f2,label\n1.0,2.0,0\n3.0,4.0,1\n")
        X, y = parser.load_data(temp_dir)
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_load_data_split_selection(self, parser, temp_dir):
        """多文件 split 选择"""
        for fname in ["train.csv", "test.csv"]:
            with open(os.path.join(temp_dir, fname), "w") as f:
                f.write("1.0,2.0,0\n3.0,4.0,1\n")
        X_train, _ = parser.load_data(temp_dir, split="train")
        X_test, _ = parser.load_data(temp_dir, split="test")
        assert X_train.shape == (2, 2)
        assert X_test.shape == (2, 2)

    def test_load_data_tsv(self, parser, temp_dir):
        """TSV 加载"""
        with open(os.path.join(temp_dir, "data.tsv"), "w") as f:
            f.write("1.0\t2.0\t0\n3.0\t4.0\t1\n")
        X, y = parser.load_data(temp_dir)
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_load_data_mixed_types(self, parser, temp_dir):
        """混合类型列编码（首行含非数值被识别为表头，剩余2行数据）"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("color,value,label\nred,1.0,0\nblue,2.0,1\n")
        X, y = parser.load_data(temp_dir)
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_class_distribution(self, parser, temp_dir):
        """类别分布统计"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
        stats = parser.parse_metadata(temp_dir)
        dist = stats["class_distribution"]
        assert dist["0"] == 2
        assert dist["1"] == 1