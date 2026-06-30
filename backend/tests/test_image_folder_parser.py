"""
图片目录解析器单元测试
覆盖: detect_format, parse_metadata, load_data, 目录钻取, 图片查找
"""
import os
import tempfile
import pytest
import numpy as np
from PIL import Image
from app.ml.datasets.image_folder_parser import ImageFolderParser
from app.ml.datasets.base_parser import DatasetParseError


class TestImageFolderParser:
    """图片目录解析器测试"""

    @pytest.fixture
    def parser(self):
        return ImageFolderParser()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def _create_dummy_image(self, path, size=(28, 28), mode="L"):
        """创建占位图像"""
        img = Image.new(mode, size, color=128)
        img.save(path)

    def _create_classified_dataset(self, root, class_names=None, images_per_class=3):
        """创建按类别分子文件夹的图片集"""
        if class_names is None:
            class_names = ["cat", "dog"]
        for cls_name in class_names:
            cls_dir = os.path.join(root, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(images_per_class):
                self._create_dummy_image(
                    os.path.join(cls_dir, f"{cls_name}_{i}.png"),
                    size=(32, 32), mode="RGB"
                )
        return root

    def test_dataset_type(self, parser):
        assert parser.dataset_type == "image_folder"

    def test_detect_format_positive(self, parser, temp_dir):
        """检测图片目录"""
        self._create_classified_dataset(temp_dir)
        assert ImageFolderParser.detect_format(temp_dir) is True

    def test_detect_format_negative(self, parser, temp_dir):
        """空目录无图片"""
        assert ImageFolderParser.detect_format(temp_dir) is False

    def test_detect_format_single_image(self, parser, temp_dir):
        """根目录直接有图片"""
        self._create_dummy_image(os.path.join(temp_dir, "img.png"))
        assert ImageFolderParser.detect_format(temp_dir) is True

    def test_detect_format_ignores_system_files(self, parser, temp_dir):
        """忽略系统文件"""
        macosx_dir = os.path.join(temp_dir, "__MACOSX")
        os.makedirs(macosx_dir)
        self._create_dummy_image(os.path.join(macosx_dir, "img.png"))
        assert ImageFolderParser.detect_format(temp_dir) is False

    def test_parse_metadata_classified(self, parser, temp_dir):
        """分类图片目录元数据"""
        self._create_classified_dataset(temp_dir, class_names=["cat", "dog", "bird"])
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 9  # 3 classes * 3 images
        assert stats["class_count"] == 3
        assert stats["dataset_type"] == "image_folder"
        dist = stats["class_distribution"]
        assert dist["cat"] == 3
        assert dist["dog"] == 3
        assert dist["bird"] == 3

    def test_parse_metadata_flat_structure(self, parser, temp_dir):
        """扁平图片目录（无类别子目录）"""
        self._create_dummy_image(os.path.join(temp_dir, "img1.png"))
        self._create_dummy_image(os.path.join(temp_dir, "img2.png"))
        stats = parser.parse_metadata(temp_dir)
        assert stats["sample_count"] == 2
        assert stats["class_count"] == 1  # 扁平结构合并为 default

    def test_parse_metadata_empty(self, parser, temp_dir):
        """空目录"""
        with pytest.raises(DatasetParseError):
            parser.parse_metadata(temp_dir)

    def test_parse_metadata_feature_shape(self, parser, temp_dir):
        """feature_shape 格式"""
        self._create_classified_dataset(temp_dir, class_names=["cat"])
        stats = parser.parse_metadata(temp_dir)
        assert "x" in stats["feature_shape"]  # e.g. "32x32x3"

    def test_parse_metadata_drill_down(self, parser, temp_dir):
        """多层根目录钻取"""
        # 模拟压缩包解压: extract/my_dataset/cat/*.png
        mid_dir = os.path.join(temp_dir, "my_dataset")
        self._create_classified_dataset(mid_dir, class_names=["cat", "dog"])
        stats = parser.parse_metadata(temp_dir)
        assert stats["class_count"] == 2
        assert stats["sample_count"] == 6

    def test_load_data_classified(self, parser, temp_dir):
        """加载分类图片数据"""
        self._create_classified_dataset(temp_dir, class_names=["cat", "dog"])
        X, y = parser.load_data(temp_dir)
        assert X.shape == (6, 32, 32, 3)  # NCHW -> NHWC
        assert y.shape == (6,)
        assert set(y.tolist()) == {0, 1}

    def test_load_data_empty(self, parser, temp_dir):
        """空目录加载"""
        with pytest.raises(DatasetParseError):
            parser.load_data(temp_dir)

    def test_load_data_mixed_extensions(self, parser, temp_dir):
        """混合图片格式"""
        cls_dir = os.path.join(temp_dir, "cat")
        os.makedirs(cls_dir)
        self._create_dummy_image(os.path.join(cls_dir, "cat1.png"))
        self._create_dummy_image(os.path.join(cls_dir, "cat2.jpg"))
        X, y = parser.load_data(temp_dir)
        assert X.shape[0] == 2

    def test_detect_format_jpeg(self, parser, temp_dir):
        """JPEG 格式"""
        os.makedirs(os.path.join(temp_dir, "class1"))
        self._create_dummy_image(os.path.join(temp_dir, "class1", "img.jpg"))
        assert ImageFolderParser.detect_format(temp_dir) is True

    def test_detect_format_bmp(self, parser, temp_dir):
        """BMP 格式"""
        os.makedirs(os.path.join(temp_dir, "class1"))
        self._create_dummy_image(os.path.join(temp_dir, "class1", "img.bmp"))
        assert ImageFolderParser.detect_format(temp_dir) is True

    def test_load_data_drill_down(self, parser, temp_dir):
        """钻取后加载"""
        mid_dir = os.path.join(temp_dir, "my_dataset")
        self._create_classified_dataset(mid_dir, class_names=["cat", "dog"])
        X, y = parser.load_data(temp_dir)
        assert X.shape[0] == 6
        assert set(y.tolist()) == {0, 1}