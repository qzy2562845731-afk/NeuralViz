"""
模型评估指标模块
支持 ROC 曲线、AUC、PR 曲线、混淆矩阵可视化等科研常用评估指标
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import io
import base64


def compute_roc_curve(
    y_true: List[int],
    y_scores: List[List[float]],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算多分类 ROC 曲线（One-vs-Rest）

    Args:
        y_true: 真实标签列表
        y_scores: 预测概率矩阵 (n_samples, n_classes)
        num_classes: 类别数
        class_names: 类别名称列表

    Returns:
        dict with keys: fpr, tpr, auc_scores, roc_data
    """
    y_true_arr = np.array(y_true, dtype=np.int64)
    y_scores_arr = np.array(y_scores, dtype=np.float64)

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i in range(num_classes):
        binary_true = (y_true_arr == i).astype(np.int64)
        binary_score = y_scores_arr[:, i] if y_scores_arr.shape[1] > i else np.zeros_like(y_true_arr)

        fpr, tpr, _ = _binary_roc_curve(binary_true, binary_score)
        auc = _auc_score(fpr, tpr)

        cls_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        fpr_dict[cls_name] = [round(float(v), 6) for v in fpr]
        tpr_dict[cls_name] = [round(float(v), 6) for v in tpr]
        auc_dict[cls_name] = round(float(auc), 4)

    macro_auc = float(np.mean(list(auc_dict.values()))) if auc_dict else 0.0

    return {
        "fpr": fpr_dict,
        "tpr": tpr_dict,
        "auc_scores": auc_dict,
        "macro_auc": round(macro_auc, 4),
        "num_classes": num_classes,
    }


def compute_pr_curve(
    y_true: List[int],
    y_scores: List[List[float]],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """计算多分类 PR 曲线（One-vs-Rest）

    Args:
        y_true: 真实标签列表
        y_scores: 预测概率矩阵
        num_classes: 类别数
        class_names: 类别名称列表

    Returns:
        dict with keys: precision, recall, ap_scores, pr_data
    """
    y_true_arr = np.array(y_true, dtype=np.int64)
    y_scores_arr = np.array(y_scores, dtype=np.float64)

    precision_dict = {}
    recall_dict = {}
    ap_dict = {}

    for i in range(num_classes):
        binary_true = (y_true_arr == i).astype(np.int64)
        binary_score = y_scores_arr[:, i] if y_scores_arr.shape[1] > i else np.zeros_like(y_true_arr)

        precision, recall, _ = _binary_pr_curve(binary_true, binary_score)
        ap = _average_precision(precision, recall)

        cls_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        precision_dict[cls_name] = [round(float(v), 6) for v in precision]
        recall_dict[cls_name] = [round(float(v), 6) for v in recall]
        ap_dict[cls_name] = round(float(ap), 4)

    macro_ap = float(np.mean(list(ap_dict.values()))) if ap_dict else 0.0

    return {
        "precision": precision_dict,
        "recall": recall_dict,
        "ap_scores": ap_dict,
        "macro_ap": round(macro_ap, 4),
        "num_classes": num_classes,
    }


def _binary_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算二分类 ROC 曲线"""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    thresholds = y_score_sorted[threshold_idxs]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    return fpr, tpr, thresholds


def _auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """使用梯形法计算 AUC"""
    if len(fpr) < 2:
        return 0.0
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    if trapz_fn is None:
        return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    return float(trapz_fn(tpr, fpr))


def _binary_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算二分类 PR 曲线"""
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    ps = tps + fps
    precision = np.divide(tps, ps, where=ps != 0, out=np.zeros_like(tps, dtype=np.float64))
    recall = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)

    precision = np.r_[1, precision]
    recall = np.r_[0, recall]
    thresholds = y_score_sorted[threshold_idxs]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    return precision, recall, thresholds


def _average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """计算平均精度 AP"""
    if len(precision) == 0:
        return 0.0
    trapz_fn = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    if trapz_fn is None:
        return float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2))
    return float(trapz_fn(precision, recall))


def plot_roc_curve_base64(
    roc_data: Dict[str, Any],
    class_names: Optional[List[str]] = None,
    dpi: int = 300,
) -> str:
    """绘制 ROC 曲线并返回 base64 编码的 PNG 图像

    Args:
        roc_data: compute_roc_curve 返回的数据
        class_names: 类别名称列表
        dpi: 分辨率（300为论文级）

    Returns:
        base64 编码的 PNG 图像
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, roc_data["num_classes"]))

    for idx, (cls_name, fpr) in enumerate(roc_data["fpr"].items()):
        tpr = roc_data["tpr"][cls_name]
        auc_val = roc_data["auc_scores"][cls_name]
        ax.plot(fpr, tpr, color=colors[idx % 10], lw=2,
                label=f'{cls_name} (AUC = {auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve (Macro AUC = {roc_data["macro_auc"]:.4f})', fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_pr_curve_base64(
    pr_data: Dict[str, Any],
    dpi: int = 300,
) -> str:
    """绘制 PR 曲线并返回 base64 编码的 PNG 图像"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)

    colors = plt.cm.tab10(np.linspace(0, 1, pr_data["num_classes"]))

    for idx, (cls_name, recall) in enumerate(pr_data["recall"].items()):
        precision = pr_data["precision"][cls_name]
        ap_val = pr_data["ap_scores"][cls_name]
        ax.plot(recall, precision, color=colors[idx % 10], lw=2,
                label=f'{cls_name} (AP = {ap_val:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve (Macro AP = {pr_data["macro_ap"]:.4f})', fontsize=14)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_confusion_matrix_base64(
    cm: List[List[int]],
    class_names: Optional[List[str]] = None,
    dpi: int = 300,
    normalize: bool = False,
) -> str:
    """绘制混淆矩阵热力图并返回 base64 编码的 PNG 图像

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        dpi: 分辨率
        normalize: 是否归一化
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm, dtype=np.float64)
    n = cm_arr.shape[0] if cm_arr.ndim == 2 else 0

    if n == 0:
        return ""

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n)]

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.divide(cm_arr, row_sums, where=row_sums != 0, out=np.zeros_like(cm_arr))

    fig, ax = plt.subplots(1, 1, figsize=(max(6, n * 0.8), max(5, n * 0.7)), dpi=dpi)
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(n),
           yticks=np.arange(n),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else '.0f'
    thresh = cm_arr.max() / 2. if cm_arr.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm_arr[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_training_curves_base64(
    metrics: List[Dict[str, Any]],
    dpi: int = 300,
) -> Dict[str, str]:
    """绘制训练曲线（Loss和Accuracy）并返回base64图像"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for m in metrics:
        epochs.append(m.get("epoch", m.get("step", 0)))
        train_loss.append(m.get("loss", 0))
        val_loss.append(m.get("val_loss", 0))
        train_acc.append(m.get("accuracy", 0))
        val_acc.append(m.get("val_accuracy", 0))

    result = {}

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5), dpi=dpi)
    ax1.plot(epochs, train_loss, 'b-', lw=2, label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', lw=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    buf1.seek(0)
    result["loss_curve"] = base64.b64encode(buf1.getvalue()).decode('utf-8')

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), dpi=dpi)
    ax2.plot(epochs, train_acc, 'b-', lw=2, label='Train Acc')
    ax2.plot(epochs, val_acc, 'r-', lw=2, label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    buf2.seek(0)
    result["acc_curve"] = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return result


def save_figure_svg(fig, dpi: int = 300) -> bytes:
    """将matplotlib图保存为SVG格式字节"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf.getvalue()


def save_figure_pdf(fig, dpi: int = 300) -> bytes:
    """将matplotlib图保存为PDF格式字节"""
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf.getvalue()
