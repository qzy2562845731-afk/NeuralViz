"""
数据增强模块
实现科研训练中常用的数据增强算法，使用numpy向量化操作保证效率
- RandomCrop: 随机裁剪（带padding）
- RandomHorizontalFlip: 随机水平翻转
- RandomVerticalFlip: 随机垂直翻转
- ColorJitter: 颜色抖动（亮度、对比度、饱和度）
- RandomErasing: 随机擦除
- CutMix: 区域级混合
- MixUp: 样本级线性插值混合
- Normalize: 标准化
- Compose: 组合增强
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import random


class Transform:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        return f"Compose([{', '.join(repr(t) for t in self.transforms)}])"


def _ensure_hwc(img: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
    """确保图像是HWC格式，返回 (img_hwc, was_chw, is_gray)

    Returns:
        img_hwc: HWC 格式的图像
        was_chw: 原始是否为 CHW 格式
        is_gray: 是否为灰度图（单通道）
    """
    if img.ndim == 2:
        return img[:, :, np.newaxis], False, True
    if img.ndim == 3:
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            return np.transpose(img, (1, 2, 0)), True, img.shape[0] == 1
        return img, False, img.shape[2] == 1
    return img, False, False


def _to_chw(img: np.ndarray, was_chw: bool, is_gray: bool, was_2d: bool = False) -> np.ndarray:
    """将HWC图像恢复到原始格式

    Args:
        img: HWC 格式图像
        was_chw: 原始是否为 CHW
        is_gray: 是否为灰度图
        was_2d: 原始是否为 2D (H,W) 无通道维
    """
    if was_2d:
        return img[:, :, 0]
    if is_gray and img.ndim == 3 and img.shape[2] == 1:
        img_2d = img[:, :, 0]
        return img_2d[np.newaxis, :, :] if was_chw else img_2d[:, :, np.newaxis]
    if was_chw:
        return np.transpose(img, (2, 0, 1))
    return img


class RandomCrop(Transform):
    def __init__(self, size: Tuple[int, int], padding: int = 4):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding

    def __call__(self, img: np.ndarray) -> np.ndarray:
        was_2d = img.ndim == 2
        img_hwc, was_chw, is_gray = _ensure_hwc(img)
        h, w, c = img_hwc.shape
        th, tw = self.size

        if self.padding > 0:
            img_hwc = np.pad(
                img_hwc,
                ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode='reflect'
            )

        ph, pw = img_hwc.shape[:2]
        if ph <= th or pw <= tw:
            return _to_chw(img_hwc, was_chw, is_gray, was_2d)

        top = random.randint(0, ph - th)
        left = random.randint(0, pw - tw)
        cropped = img_hwc[top:top+th, left:left+tw, :]
        return _to_chw(cropped, was_chw, is_gray, was_2d)

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                return img[:, :, ::-1].copy()
            elif img.ndim == 3:
                return img[:, ::-1, :].copy()
            return img[:, ::-1].copy()
        return img

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                return img[:, ::-1, :].copy()
            elif img.ndim == 3:
                return img[::-1, :, :].copy()
            return img[::-1, :].copy()
        return img

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class ColorJitter(Transform):
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        was_2d = img.ndim == 2
        img_hwc, was_chw, is_gray = _ensure_hwc(img)
        img_float = img_hwc.astype(np.float32)

        was_01 = img_float.max() <= 1.0 + 1e-6
        if not was_01:
            img_float = img_float / 255.0

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img_float = np.clip(img_float * factor, 0, 1)

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img_float.mean()
            img_float = np.clip((img_float - mean) * factor + mean, 0, 1)

        if self.saturation > 0 and img_hwc.shape[2] >= 3:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
            for c in range(3):
                img_float[:, :, c] = gray + (img_float[:, :, c] - gray) * factor
            img_float = np.clip(img_float, 0, 1)

        if not was_01:
            img_float = (img_float * 255.0).astype(img_hwc.dtype)
        else:
            img_float = img_float.astype(np.float32)

        return _to_chw(img_float, was_chw, is_gray, was_2d)

    def __repr__(self) -> str:
        return f"ColorJitter(b={self.brightness}, c={self.contrast}, s={self.saturation})"


class Normalize(Transform):
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        std_arr = np.array(std, dtype=np.float32)
        # 防止除零：将零值替换为 1.0（std=0 时不做归一化）
        std_arr = np.where(std_arr == 0, 1.0, std_arr)
        self.std = std_arr.reshape(-1, 1, 1)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            return (img - self.mean[:img.shape[0]]) / self.std[:img.shape[0]]
        elif img.ndim == 3:
            mean = self.mean.reshape(1, 1, -1)[:, :, :img.shape[2]]
            std = self.std.reshape(1, 1, -1)[:, :, :img.shape[2]]
            return (img - mean) / std
        return (img - self.mean.reshape(-1)[:img.shape[0]]) / self.std.reshape(-1)[:img.shape[0]]

    def __repr__(self) -> str:
        return f"Normalize()"


class RandomErasing(Transform):
    """随机擦除增强 (Random Erasing)

    在图像中随机选择一个矩形区域，用随机值或均值填充。
    论文: https://arxiv.org/abs/1708.04896

    Args:
        p: 应用概率
        scale: 擦除区域面积占比范围 (min, max)
        ratio: 擦除区域宽高比范围 (min, max)
        value: 填充值，'random' 表示随机值，或指定数值
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: str = "random",
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img

        was_2d = img.ndim == 2
        img_hwc, was_chw, is_gray = _ensure_hwc(img)
        h, w, c = img_hwc.shape

        area = h * w
        for _ in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if erase_h < h and erase_w < w:
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)

                if self.value == "random":
                    if img_hwc.dtype == np.uint8:
                        fill_val = np.random.randint(0, 256, (erase_h, erase_w, c)).astype(np.uint8)
                    else:
                        fill_val = np.random.rand(erase_h, erase_w, c).astype(img_hwc.dtype)
                else:
                    fill_val = np.full((erase_h, erase_w, c), self.value, dtype=img_hwc.dtype)

                img_hwc[top:top + erase_h, left:left + erase_w, :] = fill_val
                break

        return _to_chw(img_hwc, was_chw, is_gray, was_2d)

    def __repr__(self) -> str:
        return f"RandomErasing(p={self.p}, scale={self.scale}, ratio={self.ratio})"


# ============================================================
# CutMix / MixUp 是 batch-level 增强，需要同时处理图像和标签
# 在 DataLoader 层面通过 collate_fn 实现
# ============================================================

def cutmix_data(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """CutMix 数据增强 (batch-level)

    随机选取两个样本，将图像A的矩形区域替换为图像B的对应区域，
    标签按面积比例混合。

    论文: https://arxiv.org/abs/1905.04899

    Args:
        images: (N, C, H, W) 图像批次
        labels: (N,) 标签批次
        alpha: Beta 分布参数
        num_classes: 类别数

    Returns:
        mixed_images: (N, C, H, W)
        mixed_labels: (N, num_classes) one-hot 混合标签
    """
    N, C, H, W = images.shape
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    # 随机打乱索引
    rand_idx = np.random.permutation(N)

    # 计算裁剪区域
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(0, H) if H > 0 else 0
    cx = np.random.randint(0, W) if W > 0 else 0

    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    # 实际 lambda
    actual_lam = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)

    mixed_images = images.copy()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]

    # 标签混合（one-hot 编码）
    labels_onehot = np.eye(num_classes, dtype=np.float32)[labels]
    rand_labels_onehot = np.eye(num_classes, dtype=np.float32)[labels[rand_idx]]
    mixed_labels = actual_lam * labels_onehot + (1.0 - actual_lam) * rand_labels_onehot

    return mixed_images, mixed_labels


def mixup_data(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """MixUp 数据增强 (batch-level)

    随机选取两个样本，对图像和标签进行线性插值混合。

    论文: https://arxiv.org/abs/1710.09412

    Args:
        images: (N, C, H, W) 图像批次
        labels: (N,) 标签批次
        alpha: Beta 分布参数
        num_classes: 类别数

    Returns:
        mixed_images: (N, C, H, W)
        mixed_labels: (N, num_classes) one-hot 混合标签
    """
    N = images.shape[0]
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)  # 确保 lam >= 0.5

    rand_idx = np.random.permutation(N)

    mixed_images = lam * images + (1.0 - lam) * images[rand_idx]

    labels_onehot = np.eye(num_classes, dtype=np.float32)[labels]
    rand_labels_onehot = np.eye(num_classes, dtype=np.float32)[labels[rand_idx]]
    mixed_labels = lam * labels_onehot + (1.0 - lam) * rand_labels_onehot

    return mixed_images, mixed_labels


def get_augmentation_transform(augment_config: Optional[Dict[str, Any]] = None, is_train: bool = True) -> Transform:
    """根据配置构建数据增强pipeline

    Args:
        augment_config: 增强配置字典，支持字段：
            - random_crop: bool 是否启用随机裁剪
            - crop_size: int 裁剪尺寸
            - crop_padding: int 裁剪padding
            - horizontal_flip: bool 水平翻转
            - vertical_flip: bool 垂直翻转
            - flip_prob: float 翻转概率
            - color_jitter: bool 颜色抖动
            - brightness/contrast/saturation: float 颜色抖动参数
            - random_erasing: bool 是否启用随机擦除
            - erasing_prob: float 擦除概率
            - erasing_scale: list[float] 擦除面积占比范围
            - erasing_ratio: list[float] 擦除宽高比范围
            - cutmix: bool 是否启用CutMix (batch-level)
            - mixup: bool 是否启用MixUp (batch-level)
            - cutmix_alpha: float CutMix Beta分布参数
            - mixup_alpha: float MixUp Beta分布参数
        is_train: 是否为训练阶段
    """
    if not augment_config and is_train:
        return Compose([])
    if not is_train:
        return Compose([])

    transforms: List[Transform] = []

    if augment_config.get("random_crop", False):
        size = augment_config.get("crop_size", 28)
        padding = augment_config.get("crop_padding", 4)
        transforms.append(RandomCrop(size, padding=padding))

    if augment_config.get("horizontal_flip", False):
        transforms.append(RandomHorizontalFlip(p=augment_config.get("flip_prob", 0.5)))

    if augment_config.get("vertical_flip", False):
        transforms.append(RandomVerticalFlip(p=augment_config.get("flip_prob", 0.5)))

    if augment_config.get("color_jitter", False):
        transforms.append(ColorJitter(
            brightness=augment_config.get("brightness", 0.2),
            contrast=augment_config.get("contrast", 0.2),
            saturation=augment_config.get("saturation", 0.2),
        ))

    if augment_config.get("random_erasing", False):
        erasing_scale = augment_config.get("erasing_scale", [0.02, 0.33])
        erasing_ratio = augment_config.get("erasing_ratio", [0.3, 3.3])
        transforms.append(RandomErasing(
            p=augment_config.get("erasing_prob", 0.5),
            scale=tuple(erasing_scale) if isinstance(erasing_scale, list) else (0.02, 0.33),
            ratio=tuple(erasing_ratio) if isinstance(erasing_ratio, list) else (0.3, 3.3),
        ))

    return Compose(transforms)


def get_batch_augmentation_config(augment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """提取 batch-level 增强配置 (CutMix / MixUp)

    Returns:
        dict with keys: use_cutmix, cutmix_alpha, use_mixup, mixup_alpha
    """
    if not augment_config:
        return {"use_cutmix": False, "cutmix_alpha": 1.0, "use_mixup": False, "mixup_alpha": 1.0}

    return {
        "use_cutmix": augment_config.get("cutmix", False),
        "cutmix_alpha": augment_config.get("cutmix_alpha", 1.0),
        "use_mixup": augment_config.get("mixup", False),
        "mixup_alpha": augment_config.get("mixup_alpha", 1.0),
    }


def apply_augmentation_batch(X: np.ndarray, transform: Transform) -> np.ndarray:
    """对一批图像应用数据增强

    Args:
        X: 形状为 (N, C, H, W) 或 (N, H, W) 的图像批次
        transform: 增强变换

    Returns:
        增强后的图像批次，保持相同形状和dtype
    """
    if not transform.transforms:
        return X
    augmented = []
    for i in range(len(X)):
        augmented.append(transform(X[i]))
    return np.stack(augmented, axis=0)
