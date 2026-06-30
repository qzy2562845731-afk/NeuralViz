"""
JSON 数据集解析器单元测试
覆盖: JSONParser - detect_format, parse_metadata, load_data
"""
import os
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from app.ml.datasets.json_parser import JSONParser
from app.ml.datasets.base_parser import DatasetParseError


class TestJSONParser:
    """JSON 解析器"""

    @pytest.fixture
    def parser(self):
        return JSONParser()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_dataset_type(self, parser):
        assert parser.dataset_type == "json"

    def test_detect_format_positive(self, parser, temp_dir):
        """检测到 .json 文件"""
        with open(os.path.join(temp_dir, "data.json"), "w") as f:
            json.dump([[1, 2, 0], [3, 4, 1]], f)
        assert parser.detect_format(temp_dir) is True

    def test_detect_format_negative(self, parser, temp_dir):
        """无 .json 文件"""
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("1,2,0\n3,4,1\n")
        assert parser.detect_format(temp_dir) is False

    @pytest.mark.parametrize("data,expected_shape", [
        # 字典格式
        ({"data": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1]}, (2, 2)),
        ({"X": [[1.0, 2.0, 3.0]], "y": [0]}, (1, 3)),
        ({"features": [[1.0], [2.0], [3.0]], "target": [0, 1, 0]}, (3, 1)),
        # 二维数组格式
        ([[1.0, 2.0, 0], [3.0, 4.0, 1], [5.0, 6.0, 0]], (3, 2)),
        # 对象数组格式
        ([{"features": [1.0, 2.0], "label": 0}, {"features": [3.0, 4.0], "label": 1}], (2, 2)),
    ])
    def test_parse_metadata(self, parser, temp_dir, data, expected_shape):
        """解析各种 JSON 格式的元数据"""
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == expected_shape[0]
        assert stats["dataset_type"] == "json"
        assert "class_count" in stats
        assert "feature_shape" in stats
        assert "class_distribution" in stats

    def test_parse_metadata_empty_file(self, parser, temp_dir):
        """空 JSON 文件"""
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump([], f)

        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_parse_metadata_no_json_file(self, parser, temp_dir):
        """无 JSON 文件"""
        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_load_data_dict_format(self, parser, temp_dir):
        """加载字典格式数据"""
        data = {"data": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "labels": [0, 1, 0]}
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        X, y = parser.load_data(temp_dir)
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert list(y) == [0, 1, 0]

    def test_load_data_2d_array(self, parser, temp_dir):
        """加载二维数组格式"""
        data = [[1.0, 2.0, 0], [3.0, 4.0, 1], [5.0, 6.0, 0]]
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        X, y = parser.load_data(temp_dir)
        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_load_data_object_array(self, parser, temp_dir):
        """加载对象数组格式"""
        data = [
            {"features": [1.0, 2.0], "label": 0},
            {"features": [3.0, 4.0], "label": 1},
            {"features": [5.0, 6.0], "label": 0},
        ]
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        X, y = parser.load_data(temp_dir)
        assert X.shape == (3, 2)
        assert y.shape == (3,)

    def test_load_data_split_selection(self, parser, temp_dir):
        """按 split 关键词选择文件"""
        with open(os.path.join(temp_dir, "train_data.json"), "w") as f:
            json.dump({"data": [[1.0, 2.0], [3.0, 4.0]], "labels": [0, 1]}, f)
        with open(os.path.join(temp_dir, "test_data.json"), "w") as f:
            json.dump({"data": [[5.0, 6.0]], "labels": [0]}, f)

        X_train, _ = parser.load_data(temp_dir, split="train")
        X_test, _ = parser.load_data(temp_dir, split="test")
        assert X_train.shape[0] == 2
        assert X_test.shape[0] == 1

    def test_load_data_multiple_keys(self, parser, temp_dir):
        """多键字典格式"""
        data = {
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "labels": [0, 1],
            "metadata": {"version": "1.0"},
        }
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        X, y = parser.load_data(temp_dir)
        assert X.shape == (2, 2)
        assert list(y) == [0, 1]

    def test_feature_shape_format(self, parser, temp_dir):
        """特征维度格式正确"""
        data = [[1.0, 2.0, 3.0, 0], [4.0, 5.0, 6.0, 1]]
        filepath = os.path.join(temp_dir, "data.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        stats = parser.parse_metadata(temp_dir)
        assert "x" in str(stats["feature_shape"]) or str(stats["feature_shape"]).isdigit()