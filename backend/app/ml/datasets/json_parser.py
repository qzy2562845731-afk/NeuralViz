"""
JSON 格式解析器
支持 .json 文件，兼容多种 JSON 数据布局
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .base_parser import BaseDatasetParser, DatasetParseError


class JSONParser(BaseDatasetParser):
    """JSON 格式解析器

    支持以下 JSON 布局：
    1. {"data": [[feat1, feat2, ...], ...], "labels": [0, 1, ...]}
    2. {"X": [...], "y": [...]}
    3. [{"features": [...], "label": 0}, ...]  (对象数组)
    4. [[feat1, ..., label], [feat1, ..., label], ...]  (二维数组，最后一列为标签)
    """

    dataset_type = "json"

    _IGNORE_DIRS = {"__macosx", ".ds_store", "thumbs.db"}

    @staticmethod
    def detect_format(extract_path: str) -> bool:
        """检测目录下是否存在 .json 文件"""
        root = Path(extract_path)
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in JSONParser._IGNORE_DIRS):
                continue
            if name_lower.endswith(".json"):
                return True
        return False

    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析 JSON 元数据"""
        try:
            json_file = self._find_json_file(extract_path)
            if not json_file:
                raise DatasetParseError("json", "未找到 JSON 文件")

            X, y = self._read_json(str(json_file))

            sample_count = X.shape[0]
            feature_shape = "x".join(str(d) for d in X.shape[1:]) if X.ndim > 1 else str(X.shape[0])

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
            raise DatasetParseError("json", str(e))

    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载 JSON 数据"""
        try:
            json_files = self._find_json_files(extract_path)

            # 按 split 关键词选择文件
            if len(json_files) > 1:
                keywords = ["train", "training"] if split == "train" else ["test", "val", "eval"]
                for kw in keywords:
                    for f in json_files:
                        if kw in f.name.lower():
                            json_files = [f]
                            break
                    if len(json_files) == 1:
                        break

            json_file = json_files[0] if json_files else None
            if not json_file:
                raise DatasetParseError("json", "未找到 JSON 文件")

            X, y = self._read_json(str(json_file))

            if y is None:
                y = np.zeros(X.shape[0], dtype=np.int64)

            return X, y
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("json", f"加载数据失败: {str(e)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _find_json_files(self, extract_path: str) -> List[Path]:
        """查找目录下所有 JSON 文件"""
        root = Path(extract_path)
        files = []
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in self._IGNORE_DIRS):
                continue
            if name_lower.endswith(".json"):
                files.append(f)
        return sorted(files, key=lambda f: f.stat().st_size, reverse=True)

    def _find_json_file(self, extract_path: str) -> Path | None:
        """查找最大的 JSON 文件"""
        files = self._find_json_files(extract_path)
        return files[0] if files else None

    def _read_json(self, filepath: str) -> Tuple[np.ndarray, np.ndarray | None]:
        """读取 JSON 文件，返回 (X, y)

        自动识别 JSON 布局：
        - 字典格式：{"data"/"X"/"features": ..., "labels"/"y"/"target": ...}
        - 对象数组：[{...}, {...}] 格式
        - 二维数组：[[...], [...]] 格式
        """
        with open(filepath, "r", encoding="utf-8") as f:
            content = json.load(f)

        if isinstance(content, dict):
            return self._parse_dict_format(content)
        elif isinstance(content, list):
            if len(content) == 0:
                raise DatasetParseError("json", "JSON 数组为空")
            if isinstance(content[0], dict):
                return self._parse_object_array(content)
            elif isinstance(content[0], list):
                return self._parse_2d_array(content)
            elif isinstance(content[0], (int, float)):
                # 一维数组，无标签
                return np.array(content, dtype=np.float64).reshape(-1, 1), None
            else:
                raise DatasetParseError("json", f"不支持的 JSON 元素类型: {type(content[0])}")
        else:
            raise DatasetParseError("json", f"不支持的 JSON 顶层类型: {type(content)}")

    def _parse_dict_format(self, data: Dict) -> Tuple[np.ndarray, np.ndarray | None]:
        """解析字典格式: {"data": [...], "labels": [...]}"""
        X = None
        y = None

        x_keys = ["data", "X", "x", "features", "images", "inputs", "samples"]
        y_keys = ["labels", "label", "y", "target", "targets", "outputs"]

        for key in x_keys:
            if key in data:
                X = np.array(data[key], dtype=np.float64)
                break
        for key in y_keys:
            if key in data:
                y = np.array(data[key])
                break

        if X is None:
            raise DatasetParseError("json", "未找到特征数据字段，支持的键: " + ", ".join(x_keys))

        # 确保 X 是 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = y.astype(np.int64)
        else:
            y = np.zeros(X.shape[0], dtype=np.int64)

        return X, y

    def _parse_object_array(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray | None]:
        """解析对象数组格式: [{"features": [...], "label": 0}, ...]"""
        feature_keys = ["features", "data", "x", "input", "image", "values", "vector"]
        label_keys = ["label", "y", "target", "class", "category", "output"]

        # 找出特征键和标签键
        feat_key = None
        label_key = None
        for key in feature_keys:
            if key in data[0]:
                feat_key = key
                break
        for key in label_keys:
            if key in data[0]:
                label_key = key
                break

        if feat_key is None:
            # 没有标准特征键，尝试用排除法：所有非标签键的数值字段作为特征
            all_keys = list(data[0].keys())
            label_candidates = [k for k in label_keys if k in all_keys]
            if label_candidates:
                label_key = label_candidates[0]
                feat_candidates = [k for k in all_keys if k != label_key]
                if feat_candidates:
                    feat_key = feat_candidates[0]

        if feat_key is None:
            raise DatasetParseError("json", "无法识别对象数组中的特征字段")

        X_list = []
        y_list = []
        for item in data:
            feat = item.get(feat_key)
            if feat is not None:
                if isinstance(feat, (int, float)):
                    X_list.append([float(feat)])
                else:
                    X_list.append([float(v) for v in feat])
            if label_key:
                label = item.get(label_key)
                if label is not None:
                    y_list.append(label)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int64) if y_list else None

        return X, y

    def _parse_2d_array(self, data: List[List]) -> Tuple[np.ndarray, np.ndarray | None]:
        """解析二维数组格式: [[feat1, ..., label], ...]"""
        arr = np.array(data, dtype=np.float64)
        if arr.ndim < 2 or arr.shape[1] <= 1:
            return arr, None
        X = arr[:, :-1]
        y = arr[:, -1].astype(np.int64)
        return X, y