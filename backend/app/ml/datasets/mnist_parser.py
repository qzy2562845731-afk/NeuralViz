"""
MNIST IDX 二进制格式解析器
支持 .ubyte / .ubyte.gz 文件
经典手写数字数据集标准格式
"""
import os
import gzip
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .base_parser import BaseDatasetParser, DatasetParseError


class MnistIDXParser(BaseDatasetParser):
    """MNIST IDX 二进制格式解析器"""

    dataset_type = "mnist_idx"

    # 系统冗余文件/目录，解析时跳过
    _IGNORE_DIRS = {"__macosx", ".ds_store", "thumbs.db"}

    @staticmethod
    def detect_format(extract_path: str) -> bool:
        """检测目录下是否存在 MNIST IDX 格式文件

        匹配规则：同时存在 *-images-idx3-ubyte* 和 *-labels-idx1-ubyte* 文件
        """
        root = Path(extract_path)
        has_images = False
        has_labels = False

        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if name_lower.endswith((".ds_store",)):
                continue
            if "idx3-ubyte" in name_lower:
                has_images = True
            if "idx1-ubyte" in name_lower:
                has_labels = True

        return has_images and has_labels

    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析 MNIST IDX 元数据"""
        try:
            files = self._find_idx_files(extract_path)

            # 优先查找 train 分片的文件
            train_images = self._find_file_by_keyword(files, ["train", "training"], "images")
            train_labels = self._find_file_by_keyword(files, ["train", "training"], "labels")

            if not train_images:
                # 退而求其次，取第一个 images 文件
                image_files = [f for f in files if "images" in f.name.lower() or "idx3" in f.name.lower()]
                train_images = image_files[0] if image_files else None
            if not train_labels:
                label_files = [f for f in files if "labels" in f.name.lower() or "idx1" in f.name.lower()]
                train_labels = label_files[0] if label_files else None

            if not train_images or not train_labels:
                raise DatasetParseError("mnist_idx", "未找到完整的 images/labels 文件对")

            # 读取头部信息
            images_info = self._read_idx_header(str(train_images))
            labels_info = self._read_idx_header(str(train_labels))

            sample_count = images_info["num_items"]
            dims = images_info["dims"]  # [rows, cols] for images

            if len(dims) >= 2:
                rows, cols = dims[0], dims[1]
                feature_shape = f"{rows}x{cols}x1"
            elif len(dims) == 1:
                feature_shape = str(dims[0])
            else:
                feature_shape = "unknown"

            # 读取标签统计类别分布
            labels = self._read_idx_data(str(train_labels), labels_info)
            class_distribution: Dict[str, int] = {}
            for label in labels:
                key = str(int(label))
                class_distribution[key] = class_distribution.get(key, 0) + 1

            return {
                "sample_count": sample_count,
                "class_count": len(class_distribution),
                "feature_shape": feature_shape,
                "class_distribution": class_distribution,
                "dataset_type": self.dataset_type,
            }
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("mnist_idx", str(e))

    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载 MNIST 数据"""
        try:
            files = self._find_idx_files(extract_path)

            # 根据 split 选择文件
            keywords = ["train", "training"] if split == "train" else ["test", "t10k", "eval"]
            images_file = self._find_file_by_keyword(files, keywords, "images")
            labels_file = self._find_file_by_keyword(files, keywords, "labels")

            if not images_file:
                image_files = [f for f in files if "images" in f.name.lower() or "idx3" in f.name.lower()]
                images_file = image_files[0] if image_files else None
            if not labels_file:
                label_files = [f for f in files if "labels" in f.name.lower() or "idx1" in f.name.lower()]
                labels_file = label_files[0] if label_files else None

            if not images_file or not labels_file:
                raise DatasetParseError("mnist_idx", f"未找到 {split} 分片的 images/labels 文件")

            images_info = self._read_idx_header(str(images_file))
            labels_info = self._read_idx_header(str(labels_file))

            X = self._read_idx_data(str(images_file), images_info)
            y = self._read_idx_data(str(labels_file), labels_info)

            # 如果是图像数据，reshape 为 (N, H, W, 1)
            if len(images_info["dims"]) >= 2:
                X = X.reshape(-1, *images_info["dims"], 1)

            return X, y
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("mnist_idx", f"加载数据失败: {str(e)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _find_idx_files(self, extract_path: str) -> List[Path]:
        """查找目录下所有 IDX 格式文件"""
        root = Path(extract_path)
        files = []
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            # 跳过系统冗余文件
            if any(ignore in name_lower for ignore in self._IGNORE_DIRS):
                continue
            if "idx" in name_lower and "ubyte" in name_lower:
                files.append(f)
        return files

    def _find_file_by_keyword(
        self, files: List[Path], keywords: List[str], file_type: str
    ) -> Path | None:
        """根据关键词查找文件"""
        type_keyword = "images" if file_type == "images" else "labels"
        idx_keyword = "idx3" if file_type == "images" else "idx1"

        # 先按 split 关键词匹配
        for f in files:
            name_lower = f.name.lower()
            if any(kw in name_lower for kw in keywords):
                if type_keyword in name_lower or idx_keyword in name_lower:
                    return f
        return None

    def _open_maybe_gz(self, filepath: str):
        """打开文件，自动处理 .gz 压缩"""
        if filepath.endswith(".gz"):
            return gzip.open(filepath, "rb")
        return open(filepath, "rb")

    def _read_idx_header(self, filepath: str) -> Dict[str, Any]:
        """读取 IDX 文件头部信息

        IDX 格式：
        - magic number (4 bytes): 前2字节为0，第3字节为数据类型，第4字节为维度数
        - num_items (4 bytes): 数据项数量
        - dims: 各维度大小（每个4 bytes）
        """
        with self._open_maybe_gz(filepath) as f:
            # 读取 magic number
            magic = struct.unpack(">I", f.read(4))[0]
            data_type = (magic >> 8) & 0xFF
            num_dims = magic & 0xFF

            # 数据类型映射
            dtype_map = {
                0x08: np.uint8,
                0x09: np.int8,
                0x0B: np.int16,
                0x0C: np.int32,
                0x0D: np.float32,
                0x0E: np.float64,
            }
            dtype = dtype_map.get(data_type, np.uint8)

            # 读取维度信息
            if num_dims == 0:
                num_items = struct.unpack(">I", f.read(4))[0]
                return {"dtype": dtype, "num_items": num_items, "dims": []}

            num_items = struct.unpack(">I", f.read(4))[0]
            dims = []
            for _ in range(num_dims - 1):
                dims.append(struct.unpack(">I", f.read(4))[0])

            return {"dtype": dtype, "num_items": num_items, "dims": dims}

    def _read_idx_data(self, filepath: str, header: Dict[str, Any]) -> np.ndarray:
        """读取 IDX 文件数据部分"""
        with self._open_maybe_gz(filepath) as f:
            # 跳过头部：magic(4) + num_items(4) + dims(4 * (num_dims-1))
            num_dims = (struct.unpack(">I", f.read(4))[0]) & 0xFF
            f.read(4)  # num_items
            f.read(4 * (num_dims - 1))  # dims

            # 读取数据
            total_items = header["num_items"]
            total_elements = total_items
            for d in header["dims"]:
                total_elements *= d

            data = np.frombuffer(f.read(total_elements * np.dtype(header["dtype"]).itemsize), dtype=header["dtype"])

            return data
