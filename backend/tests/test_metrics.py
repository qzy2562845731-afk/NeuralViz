"""
模型评估指标单元测试
覆盖: compute_roc_curve, compute_pr_curve, 内部辅助函数
"""
import pytest
import numpy as np
from app.ml.metrics import (
    compute_roc_curve, compute_pr_curve,
    _binary_roc_curve, _binary_pr_curve,
    _auc_score, _average_precision,
)


class TestROCCurve:
    """ROC 曲线测试"""

    @pytest.fixture
    def sample_data(self):
        """生成模拟多分类数据"""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, n_samples).tolist()
        y_scores = np.random.rand(n_samples, n_classes).tolist()
        # 归一化使每行和为 1
        y_scores = (np.array(y_scores) / np.array(y_scores).sum(axis=1, keepdims=True)).tolist()
        return y_true, y_scores, n_classes

    def test_basic_roc(self, sample_data):
        """基础 ROC 曲线计算"""
        y_true, y_scores, n_classes = sample_data
        result = compute_roc_curve(y_true, y_scores, n_classes)
        assert "fpr" in result
        assert "tpr" in result
        assert "auc_scores" in result
        assert "macro_auc" in result
        assert result["num_classes"] == n_classes
        assert len(result["fpr"]) == n_classes

    def test_auc_range(self, sample_data):
        """AUC 值在 [0, 1] 范围内"""
        y_true, y_scores, n_classes = sample_data
        result = compute_roc_curve(y_true, y_scores, n_classes)
        for auc_val in result["auc_scores"].values():
            assert 0.0 <= auc_val <= 1.0
        assert 0.0 <= result["macro_auc"] <= 1.0

    def test_with_class_names(self, sample_data):
        """带类别名称的 ROC"""
        y_true, y_scores, n_classes = sample_data
        class_names = ["cat", "dog", "bird"]
        result = compute_roc_curve(y_true, y_scores, n_classes, class_names)
        for name in class_names:
            assert name in result["auc_scores"]

    def test_perfect_classification(self):
        """完美分类的 ROC"""
        y_true = [0, 0, 1, 1, 2, 2]
        y_scores = [
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.05, 0.9],
        ]
        result = compute_roc_curve(y_true, y_scores, 3)
        for auc_val in result["auc_scores"].values():
            assert auc_val >= 0.9  # 接近完美


class TestPRCurve:
    """PR 曲线测试"""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        n_classes = 4
        y_true = np.random.randint(0, n_classes, n_samples).tolist()
        y_scores = np.random.rand(n_samples, n_classes)
        y_scores = (y_scores / y_scores.sum(axis=1, keepdims=True)).tolist()
        return y_true, y_scores, n_classes

    def test_basic_pr(self, sample_data):
        """基础 PR 曲线计算"""
        y_true, y_scores, n_classes = sample_data
        result = compute_pr_curve(y_true, y_scores, n_classes)
        assert "precision" in result
        assert "recall" in result
        assert "ap_scores" in result
        assert "macro_ap" in result
        assert result["num_classes"] == n_classes

    def test_ap_range(self, sample_data):
        """AP 值在 [0, 1] 范围内"""
        y_true, y_scores, n_classes = sample_data
        result = compute_pr_curve(y_true, y_scores, n_classes)
        for ap_val in result["ap_scores"].values():
            assert 0.0 <= ap_val <= 1.0


class TestBinaryROCCurve:
    """二分类 ROC 内部函数"""

    def test_binary_roc_perfect(self):
        """完美分类"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        fpr, tpr, thresholds = _binary_roc_curve(y_true, y_score)
        auc = _auc_score(fpr, tpr)
        assert auc == 1.0

    def test_binary_roc_random(self):
        """随机分类"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_score = np.random.rand(100)
        fpr, tpr, _ = _binary_roc_curve(y_true, y_score)
        auc = _auc_score(fpr, tpr)
        assert 0.0 <= auc <= 1.0

    def test_binary_roc_inverse(self):
        """完全错误分类"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.1, 0.2])
        fpr, tpr, _ = _binary_roc_curve(y_true, y_score)
        auc = _auc_score(fpr, tpr)
        assert auc == 0.0

    def test_auc_single_sample(self):
        """单样本 AUC（梯形面积）"""
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc = _auc_score(fpr, tpr)
        # 梯形面积 = (1-0) * (0+1)/2 = 0.5
        assert auc == 0.5


class TestBinaryPRCurve:
    """二分类 PR 内部函数"""

    def test_binary_pr_perfect(self):
        """完美分类 AP"""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        precision, recall, _ = _binary_pr_curve(y_true, y_score)
        ap = _average_precision(precision, recall)
        assert ap == 1.0

    def test_average_precision(self):
        """平均精度计算"""
        precision = np.array([1.0, 0.8, 0.6, 0.0])
        recall = np.array([0.0, 0.5, 0.8, 1.0])
        ap = _average_precision(precision, recall)
        assert 0.0 <= ap <= 1.0

    def test_empty_precision(self):
        """空 precision 数组"""
        ap = _average_precision(np.array([]), np.array([]))
        assert ap == 0.0