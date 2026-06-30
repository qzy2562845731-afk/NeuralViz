"""
CSV/TSV 表格格式解析器
支持 .csv / .tsv 文件
"""
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .base_parser import BaseDatasetParser, DatasetParseError


class CSVParser(BaseDatasetParser):
    """CSV/TSV 表格格式解析器"""

    dataset_type = "csv"

    # 系统冗余文件
    _IGNORE_DIRS = {"__macosx", ".ds_store", "thumbs.db"}

    @staticmethod
    def detect_format(extract_path: str) -> bool:
        """检测目录下是否存在 .csv 或 .tsv 文件"""
        root = Path(extract_path)
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in CSVParser._IGNORE_DIRS):
                continue
            if name_lower.endswith((".csv", ".tsv")):
                return True
        return False

    def parse_metadata(self, extract_path: str) -> Dict[str, Any]:
        """解析 CSV 元数据"""
        try:
            csv_file = self._find_largest_csv(extract_path)
            if not csv_file:
                raise DatasetParseError("csv", "未找到 CSV/TSV 文件")

            delimiter = "\t" if csv_file.suffix.lower() == ".tsv" else ","
            X, y, has_header = self._read_csv(str(csv_file), delimiter)

            sample_count = X.shape[0]
            feature_shape = str(X.shape[1]) if X.ndim > 1 else "1"

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
            raise DatasetParseError("csv", str(e))

    def load_data(self, extract_path: str, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """加载 CSV 数据"""
        try:
            csv_files = self._find_csv_files(extract_path)

            # 按 split 关键词选择文件
            if len(csv_files) > 1:
                keywords = ["train", "training"] if split == "train" else ["test", "val", "eval"]
                for kw in keywords:
                    for f in csv_files:
                        if kw in f.name.lower():
                            csv_files = [f]
                            break
                    if len(csv_files) == 1:
                        break

            csv_file = csv_files[0] if csv_files else None
            if not csv_file:
                raise DatasetParseError("csv", "未找到 CSV/TSV 文件")

            delimiter = "\t" if csv_file.suffix.lower() == ".tsv" else ","
            X, y, _ = self._read_csv(str(csv_file), delimiter)

            if y is None:
                y = np.zeros(X.shape[0], dtype=np.int64)

            return X, y
        except DatasetParseError:
            raise
        except Exception as e:
            raise DatasetParseError("csv", f"加载数据失败: {str(e)}")

    # ============================================================
    # 内部工具方法
    # ============================================================

    def _find_csv_files(self, extract_path: str) -> List[Path]:
        """查找目录下所有 CSV/TSV 文件"""
        root = Path(extract_path)
        files = []
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if any(ignore in name_lower for ignore in self._IGNORE_DIRS):
                continue
            if name_lower.endswith((".csv", ".tsv")):
                files.append(f)
        return files

    def _find_largest_csv(self, extract_path: str) -> Path | None:
        """查找最大的 CSV/TSV 文件"""
        files = self._find_csv_files(extract_path)
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_size)

    def _read_csv(self, filepath: str, delimiter: str) -> Tuple[np.ndarray, np.ndarray | None, bool]:
        """读取 CSV 文件，返回 (X, y, has_header)

        约定：默认最后一列为标签列，其余为特征列
        首行自动识别是否为表头
        """
        rows = []
        with open(filepath, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        if not rows:
            raise DatasetParseError("csv", "CSV 文件为空")

        # 检测表头：如果第一行包含非数值字符串，则认为是表头
        has_header = self._detect_header(rows[0])
        if has_header:
            rows = rows[1:]

        if not rows:
            raise DatasetParseError("csv", "CSV 文件无有效数据行")

        # 尝试转换为数值数组
        try:
            data = np.array(rows, dtype=np.float64)
        except ValueError:
            # 混合类型，尝试逐列转换
            data = self._parse_mixed_types(rows)

        if data.ndim < 2:
            # 单列数据
            return data, None, has_header

        X = data[:, :-1]
        y = data[:, -1]
        return X, y, has_header

    def _detect_header(self, first_row: List[str]) -> bool:
        """检测首行是否为表头"""
        for cell in first_row:
            cell = cell.strip()
            if not cell:
                continue
            try:
                float(cell)
            except ValueError:
                # 非数值单元格，认为是表头
                return True
        return False

    def _parse_mixed_types(self, rows: List[List[str]]) -> np.ndarray:
        """解析混合类型数据，将非数值列编码为数值"""
        if not rows:
            return np.array([])

        num_cols = max(len(row) for row in rows)
        # 对齐行长度
        for row in rows:
            while len(row) < num_cols:
                row.append("")

        # 逐列处理
        col_data = [[] for _ in range(num_cols)]
        col_encoders = [None] * num_cols

        for col_idx in range(num_cols):
            values = [row[col_idx] for row in rows]
            try:
                # 尝试转为 float
                col_data[col_idx] = [float(v) for v in values]
            except ValueError:
                # 分类编码
                unique_vals = list(set(values))
                mapping = {v: i for i, v in enumerate(unique_vals)}
                col_data[col_idx] = [mapping[v] for v in values]
                col_encoders[col_idx] = mapping

        return np.array(col_data, dtype=np.float64).T
