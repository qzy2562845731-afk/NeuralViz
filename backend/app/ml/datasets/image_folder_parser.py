"""
图片目录格式解析器
原有逻辑迁移 + 增强兼容
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .base_parser import BaseDatasetParser, DatasetParseError


class ImageFolderParser(BaseDatasetParser):
    """图片目录格式解析器

    识别标准图像分类目录结构：根目录下按类别分子文件夹
    增强兼容：
    - 自动向下递归遍历，兼容压缩包内多一层根目录
    - 自动过滤 __MACOSX、.DS_Store、Thumbs.db 等系统冗余文件
    - 支持 jpg、jpeg、png、bmp、webp 常见图片格式
    """

    dataset_type = "image_folder"

    # 支持的图片扩展名
    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 系统冗余文件/目录，解析时跳过
    _IGNORE_NAMES = {"__macosx", ".ds_store", "thumbs.db", "desktop.ini"}

    @staticmethod
    def detect_format(extract_path: str) -> bool:
        """检测目录下是否存在按类别分子文件夹的图片集"""
        root = Path(extract_path)

        # 查找有效图片文件
        all_images = ImageFolderParser._find_images(root)
        return len(all_images) > 0

    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析图片目录元数据"""
        try:
            root = Path(extract_path)
            all_images = self._find_images(root)

            if not all_images:
                raise DatasetParseError("image_folder", "未找到任何图片文件")

            # 检测是否有多层根目录：如果所有图片都在一个共同的第一级子目录下，
            # 且该子目录下还有子目录（类别目录），则向下钻取一层
            root = self._drill_down_to_class_root(root, all_images)

            # 重新查找图片（基于钻取后的根目录）
            all_images = self._find_images(root)

            # 按类别分组：第一级有效子目录名作为类别名
            class_distribution: Dict[str, int] = {}

            for img_path in all_images:
                rel = img_path.relative_to(root)
                parts = rel.parts

                if len(parts) > 1:
                    # 有子目录，第一级目录名作为类别
                    class_name = parts[0]
                else:
                    # 根目录下的文件，无类别
                    class_name = "unclassified"

                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

            # 如果只有 unclassified 一个类别，说明是扁平结构
            if len(class_distribution) == 1 and "unclassified" in class_distribution:
                count = class_distribution["unclassified"]
                class_distribution = {"default": count}

            # 读取第一张图像的尺寸
            image_size = self._read_image_size(str(all_images[0]))

            return {
                "sample_count": len(all_images),
                "class_count": len(class_distribution),
                "feature_shape": image_size,
                "class_distribution": class_distribution,
                "dataset_type": self.dataset_type,
            }
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("image_folder", str(e))

    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载图片数据"""
        try:
            from PIL import Image

            root = Path(extract_path)
            all_images = self._find_images(root)

            if not all_images:
                raise DatasetParseError("image_folder", "未找到任何图片文件")

            # 钻取到类别根目录
            root = self._drill_down_to_class_root(root, all_images)
            all_images = self._find_images(root)

            # 按类别分组
            class_names = sorted(set(
                img.relative_to(root).parts[0] if len(img.relative_to(root).parts) > 1
                else "default"
                for img in all_images
            ))
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            images = []
            labels = []

            for img_path in all_images:
                try:
                    rel = img_path.relative_to(root)
                    parts = rel.parts
                    class_name = parts[0] if len(parts) > 1 else "default"

                    with Image.open(str(img_path)) as img:
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(class_to_idx[class_name])
                except Exception:
                    # 单个文件解析失败不中断整体流程
                    continue

            if not images:
                raise DatasetParseError("image_folder", "所有图片文件均无法读取")

            X = np.array(images)
            y = np.array(labels, dtype=np.int64)
            return X, y
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("image_folder", f"加载数据失败: {str(e)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    @staticmethod
    def _find_images(root: Path) -> List[Path]:
        """查找目录下所有有效图片文件

        - 递归遍历，兼容多层级目录
        - 过滤系统冗余文件
        - 去重
        """
        all_images = []

        for ext in ImageFolderParser._IMAGE_EXTENSIONS:
            all_images.extend(root.rglob(f"*{ext}"))
            all_images.extend(root.rglob(f"*{ext.upper()}"))

        # 过滤系统冗余文件 + 去重
        seen = set()
        unique_images = []
        for img_path in all_images:
            # 检查路径中是否包含冗余目录名
            path_parts_lower = [p.lower() for p in img_path.parts]
            if any(ignore in path_parts_lower for ignore in ImageFolderParser._IGNORE_NAMES):
                continue

            resolved = str(img_path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_images.append(img_path)

        return unique_images

    def _drill_down_to_class_root(self, root: Path, all_images: List[Path]) -> Path:
        """检测并钻取多层根目录

        压缩包内常有多余的根目录（如 my_dataset/cat/*.jpg），
        此时 my_dataset 会被误识别为单一类别。该方法检测：
        如果所有图片都位于同一个第一级子目录下，且该子目录下还存在
        子目录（真正的类别目录），则将根目录向下钻取一层。
        重复此过程直到不再需要钻取。

        示例：
            extract_path/my_dataset/cat/*.jpg   ┐
            extract_path/my_dataset/dog/*.jpg   ┘ → 钻取到 extract_path/my_dataset/

        Args:
            root: 当前根目录
            all_images: 所有图片路径列表（相对于 root 解析）

        Returns:
            钻取后的根目录（无需钻取时返回原 root）
        """
        current_root = root

        while True:
            # 收集每张图片相对于当前根目录的第一级目录名
            first_parts = set()
            all_in_subdir = True

            for img_path in all_images:
                try:
                    rel = img_path.relative_to(current_root)
                    parts = rel.parts
                    if len(parts) <= 1:
                        # 图片直接位于根目录下，无需钻取
                        all_in_subdir = False
                        break
                    first_parts.add(parts[0])
                except ValueError:
                    # 路径不在当前根目录下，停止钻取
                    all_in_subdir = False
                    break

            # 有图片在根目录下，或第一级目录不唯一，则无需钻取
            if not all_in_subdir or len(first_parts) != 1:
                break

            # 所有图片都在同一个第一级子目录下
            single_dir_name = next(iter(first_parts))
            single_dir_path = current_root / single_dir_name

            if not single_dir_path.is_dir():
                break

            # 检查该子目录下是否还有子目录（真正的类别目录）
            has_subdirs = any(
                child.is_dir() and child.name.lower() not in self._IGNORE_NAMES
                for child in single_dir_path.iterdir()
            )

            if not has_subdirs:
                # 该子目录下没有子目录，说明它本身就是类别目录，不再钻取
                break

            # 钻取一层，并基于新根目录重新查找图片
            current_root = single_dir_path
            all_images = self._find_images(current_root)

            # 如果钻取后没有图片了，回退到上一层
            if not all_images:
                break

        return current_root

    def _read_image_size(self, image_path: str) -> str:
        """读取图像尺寸，返回格式如 '28x28x1'"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                w, h = img.size
                mode_to_channels = {"L": 1, "RGB": 3, "RGBA": 4, "P": 3, "CMYK": 4}
                c = mode_to_channels.get(img.mode, 3)
                return f"{w}x{h}x{c}"
        except Exception:
            return "unknown"
