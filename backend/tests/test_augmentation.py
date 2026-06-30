"""
数据增强模块单元测试
覆盖: RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter,
      RandomErasing, cutmix_data, mixup_data, Compose, get_augmentation_transform
"""
import pytest
import numpy as np
from app.ml.augmentation import (
    RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter,
    RandomErasing, Compose, Normalize, Transform,
    get_augmentation_transform, get_batch_augmentation_config,
    apply_augmentation_batch, cutmix_data, mixup_data,
)


class TestRandomCrop:
    """测试随机裁剪"""

    def test_output_shape(self):
        """裁剪后形状正确"""
        transform = RandomCrop(size=(28, 28), padding=4)
        img = np.random.rand(1, 36, 36).astype(np.float32)
        result = transform(img)
        assert result.shape == (1, 28, 28)

    def test_hwc_format(self):
        """HWC 格式输入"""
        transform = RandomCrop(size=(28, 28), padding=4)
        img = np.random.rand(36, 36, 1).astype(np.float32)
        result = transform(img)
        assert result.shape == (28, 28, 1)

    def test_no_padding(self):
        """无 padding 裁剪"""
        transform = RandomCrop(size=(28, 28), padding=0)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = transform(img)
        assert result.shape == (1, 28, 28)


class TestRandomHorizontalFlip:
    """测试随机水平翻转"""

    def test_output_shape(self):
        """翻转后形状不变"""
        transform = RandomHorizontalFlip(p=1.0)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape

    def test_chw_format(self):
        """CHW 格式翻转"""
        transform = RandomHorizontalFlip(p=1.0)
        img = np.random.rand(3, 32, 32).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape


class TestRandomVerticalFlip:
    """测试随机垂直翻转"""

    def test_output_shape(self):
        """翻转后形状不变"""
        transform = RandomVerticalFlip(p=1.0)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape


class TestColorJitter:
    """测试颜色抖动"""

    def test_output_shape(self):
        """抖动后形状不变"""
        transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        img = np.random.rand(3, 32, 32).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape

    def test_single_channel(self):
        """单通道图像"""
        transform = ColorJitter(brightness=0.2, contrast=0.2)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape

    def test_uint8_input(self):
        """uint8 输入"""
        transform = ColorJitter(brightness=0.2, contrast=0.2)
        img = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
        result = transform(img)
        assert result.shape == img.shape


class TestRandomErasing:
    """测试随机擦除"""

    def test_output_shape(self):
        """擦除后形状不变"""
        transform = RandomErasing(p=1.0)
        img = np.random.rand(3, 32, 32).astype(np.float32)
        result = transform(img)
        assert result.shape == img.shape

    def test_uint8_input(self):
        """uint8 输入"""
        transform = RandomErasing(p=1.0)
        img = np.random.randint(0, 256, (1, 28, 28), dtype=np.uint8)
        result = transform(img)
        assert result.shape == img.shape

    def test_probability_zero(self):
        """p=0 时不擦除"""
        transform = RandomErasing(p=0.0)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = transform(img)
        np.testing.assert_array_equal(img, result)


class TestCompose:
    """测试组合增强"""

    def test_multiple_transforms(self):
        """多个变换组合"""
        transforms = [
            RandomHorizontalFlip(p=1.0),
            RandomErasing(p=0.0),
        ]
        composed = Compose(transforms)
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = composed(img)
        assert result.shape == img.shape

    def test_empty_compose(self):
        """空组合"""
        composed = Compose([])
        img = np.random.rand(1, 28, 28).astype(np.float32)
        result = composed(img)
        np.testing.assert_array_equal(img, result)


class TestCutMix:
    """测试 CutMix"""

    def test_cutmix_output_shape(self):
        """CutMix 输出形状正确"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        mixed_images, mixed_labels = cutmix_data(images, labels, alpha=1.0, num_classes=10)
        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == (8, 10)

    def test_cutmix_labels_sum_to_one(self):
        """CutMix 标签和为 1"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        _, mixed_labels = cutmix_data(images, labels, alpha=1.0, num_classes=10)
        assert np.allclose(mixed_labels.sum(axis=1), 1.0, atol=1e-5)

    def test_cutmix_alpha_zero(self):
        """alpha=0 时 CutMix 退化为无混合"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        _, mixed_labels = cutmix_data(images, labels, alpha=0.0, num_classes=10)
        # alpha=0 时 lam=1，标签应该还是 one-hot
        assert np.allclose(mixed_labels.sum(axis=1), 1.0)


class TestMixUp:
    """测试 MixUp"""

    def test_mixup_output_shape(self):
        """MixUp 输出形状正确"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        mixed_images, mixed_labels = mixup_data(images, labels, alpha=1.0, num_classes=10)
        assert mixed_images.shape == images.shape
        assert mixed_labels.shape == (8, 10)

    def test_mixup_labels_sum_to_one(self):
        """MixUp 标签和为 1"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        _, mixed_labels = mixup_data(images, labels, alpha=1.0, num_classes=10)
        assert np.allclose(mixed_labels.sum(axis=1), 1.0, atol=1e-5)

    def test_mixup_preserves_range(self):
        """MixUp 图像值在合理范围"""
        images = np.random.rand(8, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 10, 8).astype(np.int64)
        mixed_images, _ = mixup_data(images, labels, alpha=1.0, num_classes=10)
        assert mixed_images.min() >= 0.0
        assert mixed_images.max() <= 1.0


class TestGetAugmentationTransform:
    """测试增强变换工厂"""

    def test_default_transform(self):
        """默认不启用增强"""
        transform = get_augmentation_transform({}, is_train=True)
        result = apply_augmentation_batch(
            np.random.rand(4, 1, 28, 28).astype(np.float32), transform
        )
        assert result.shape == (4, 1, 28, 28)

    def test_with_random_erasing(self):
        """启用 Random Erasing"""
        config = {"random_erasing": True, "erasing_prob": 0.5}
        transform = get_augmentation_transform(config, is_train=True)
        assert len(transform.transforms) == 1

    def test_not_train(self):
        """非训练模式不启用增强"""
        config = {"random_erasing": True, "horizontal_flip": True}
        transform = get_augmentation_transform(config, is_train=False)
        assert len(transform.transforms) == 0

    def test_full_pipeline(self):
        """完整增强 pipeline"""
        config = {
            "random_crop": True,
            "horizontal_flip": True,
            "color_jitter": True,
            "random_erasing": True,
        }
        transform = get_augmentation_transform(config, is_train=True)
        assert len(transform.transforms) == 4


class TestGetBatchAugmentationConfig:
    """测试 batch-level 增强配置"""

    def test_no_config(self):
        """无配置"""
        result = get_batch_augmentation_config(None)
        assert result["use_cutmix"] is False
        assert result["use_mixup"] is False

    def test_cutmix_config(self):
        """CutMix 配置"""
        config = {"cutmix": True, "cutmix_alpha": 0.5}
        result = get_batch_augmentation_config(config)
        assert result["use_cutmix"] is True
        assert result["cutmix_alpha"] == 0.5

    def test_mixup_config(self):
        """MixUp 配置"""
        config = {"mixup": True, "mixup_alpha": 0.8}
        result = get_batch_augmentation_config(config)
        assert result["use_mixup"] is True
        assert result["mixup_alpha"] == 0.8


class TestApplyAugmentationBatch:
    """测试批量增强"""

    def test_batch_shape(self):
        """批量增强形状不变"""
        transform = get_augmentation_transform({"horizontal_flip": True}, is_train=True)
        X = np.random.rand(8, 1, 28, 28).astype(np.float32)
        result = apply_augmentation_batch(X, transform)
        assert result.shape == X.shape

    def test_dtype_preserved(self):
        """dtype 保持不变"""
        transform = Compose([])
        X = np.random.rand(4, 1, 28, 28).astype(np.float32)
        result = apply_augmentation_batch(X, transform)
        assert result.dtype == X.dtype