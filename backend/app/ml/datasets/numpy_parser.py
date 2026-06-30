"""
NumPy 数组格式解析器
支持 .npy / .npz 文件
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .base_parser import BaseDatasetParser, DatasetParseError


class NumpyParser(BaseDatasetParser):
    """NumPy 数组格式解析器"""

    dataset_type = "numpy"

    # 系统冗余文件
    _IGNORE_DIRS = {"__macosx", ".ds_store", "thumbs.db"}

    @staticmethod
    def detect_format(extract_path: str) -> bool:
        """检测目录下是否存在 .npy 或 .npz 文件"""
        root = Path(extract_path)
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in NumpyParser._IGNORE_DIRS):
                continue
            if name_lower.endswith((".npy", ".npz")):
                return True
        return False

    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析 NumPy 数组元数据"""
        try:
            npz_files, npy_files = self._find_numpy_files(extract_path)

            X, y = self._load_arrays(npz_files, npy_files)

            sample_count = X.shape[0]
            feature_shape = "x".join(str(d) for d in X.shape[1:]) if X.ndim > 1 else str(X.shape[0])

            # 统计类别分布
            class_distribution: Dict[str, int] = {}
            if y is not None:
                for label in y:
                    key = str(int(label)) if np.issubdtype(y.dtype, np.number) else str(label)
                    class_distribution[key] = class_distribution.get(key, 0) + 1

            class_count = len(class_distribution) if class_distribution else 0

            return {
                "sample_count": sample_count,
                "class_count": class_count,
                "feature_shape": feature_shape,
                "class_distribution": class_distribution,
                "dataset_type": self.dataset_type,
            }
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("numpy", str(e))

    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载 NumPy 数据"""
        try:
            npz_files, npy_files = self._find_numpy_files(extract_path)

            # 如果有多个 npz 文件，按 split 关键词选择
            if len(npz_files) > 1:
                keywords = ["train", "training"] if split == "train" else ["test", "val", "eval"]
                for kw in keywords:
                    for f in npz_files:
                        if kw in f.name.lower():
                            npz_files = [f]
                            break
                    if len(npz_files) == 1:
                        break

            X, y = self._load_arrays(npz_files, npy_files)

            if y is None:
                y = np.zeros(X.shape[0], dtype=np.int64)

            return X, y
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("numpy", f"加载数据失败: {str(e)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _find_numpy_files(self, extract_path: str) -> Tuple[List[Path], List[Path]]:
        """查找目录下所有 .npz 和 .npy 文件"""
        root = Path(extract_path)
        npz_files = []
        npy_files = []

        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in self._IGNORE_DIRS):
                continue
            if name_lower.endswith(".npz"):
                npz_files.append(f)
            elif name_lower.endswith(".npy"):
                npy_files.append(f)

        return npz_files, npy_files

    def _load_arrays(self, npz_files: List[Path], npy_files: List[Path]) -> Tuple[np.ndarray, np.ndarray | None]:
        """从文件加载数组

        约定规则：
        - .npz 优先，默认键名 X/data 为特征，y/label 为标签
        - 单 .npy 文件默认最后一列为标签
        """
        # 优先使用 .npz 文件
        if npz_files:
            npz_file = npz_files[0]  # 取第一个
            data = np.load(str(npz_file), allow_pickle=True)

            X = None
            y = None

            # 尝试常见键名
            x_keys = ["x", "data", "features", "images", "train_x", "train_data"]
            y_keys = ["y", "label", "labels", "target", "targets", "train_y", "train_labels"]

            if isinstance(data, np.lib.npyio.NpzFile):
                available_keys = list(data.keys())
                for key in x_keys:
                    if key.lower() in [k.lower() for k in available_keys]:
                        for k in available_keys:
                            if k.lower() == key.lower():
                                X = data[k]
                                break
                        break
                for key in y_keys:
                    if key.lower() in [k.lower() for k in available_keys]:
                        for k in available_keys:
                            if k.lower() == key.lower():
                                y = data[k]
                                break
                        break

                # 如果没找到标准键名，取第一个数组作为 X
                if X is None and available_keys:
                    X = data[available_keys[0]]
                    # 如果有第二个数组，作为 y
                    if len(available_keys) > 1:
                        y = data[available_keys[1]]
            else:
                X = data

            if X is None:
                raise DatasetParseError("numpy", f"NPZ 文件中未找到有效数组，可用键: {list(data.keys()) if isinstance(data, np.lib.npyio.NpzFile) else 'N/A'}")

            return X, y

        # 使用 .npy 文件
        if npy_files:
            # 如果有多个 .npy 文件，尝试按名称匹配 X/y
            if len(npy_files) >= 2:
                x_file = None
                y_file = None
                x_keywords = ["x", "data", "features", "images"]
                y_keywords = ["y", "label", "labels", "target"]

                for f in npy_files:
                    name_lower = f.stem.lower()
                    if any(kw in name_lower for kw in x_keywords) and x_file is None:
                        x_file = f
                    if any(kw in name_lower for kw in y_keywords) and y_file is None:
                        y_file = f

                if x_file and y_file:
                    X = np.load(str(x_file), allow_pickle=True)
                    y = np.load(str(y_file), allow_pickle=True)
                    return X, y

            # 单个 .npy 文件：最后一列为标签
            npy_file = npy_files[0]
            data = np.load(str(npy_file), allow_pickle=True)

            if data.ndim < 2:
                # 一维数组，无标签
                return data, None

            X = data[:, :-1]
            y = data[:, -1]
            return X, y

        raise DatasetParseError("numpy", "未找到 .npy 或 .npz 文件")
