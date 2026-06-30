"""
损失函数模块
提供多样化的损失函数，支持分类/回归/序列任务：
- CrossEntropyLoss: 标准交叉熵
- FocalLoss: 聚焦损失，处理类别不平衡
- LabelSmoothingCrossEntropy: 标签平滑交叉熵
- DiceLoss: Dice 损失（常用于分割，也可用于分类）
- CombinedLoss: 组合损失，支持多损失加权组合
- TripletLoss: 三元组损失
- ContrastiveLoss: 对比损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance
    聚焦损失：降低易分类样本的权重，聚焦难分类样本
    arXiv: https://arxiv.org/abs/1708.02002

    Args:
        alpha: 类别权重，可为 float 或 list[float]
        gamma: 聚焦参数，越大越关注难分类样本
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss
    标签平滑：将硬标签转换为软标签，防止过拟合，提高泛化能力
    arXiv: https://arxiv.org/abs/1906.02629

    Args:
        smoothing: 平滑系数，0 表示无平滑，0.1 表示 10% 平滑
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # 构造平滑标签
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_labels * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Dice Loss for multi-class classification
    基于 Dice 系数的损失函数，对类别不平衡鲁棒

    Args:
        smooth: 平滑因子，防止除零
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        probs = F.softmax(inputs, dim=-1)
        targets_one_hot = F.one_hot(targets, n_classes).float()

        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """组合损失：支持多个损失函数加权组合

    Args:
        losses: 损失函数字典，key 为名称，value 为 (loss_fn, weight)
        reduction: 最终损失的 reduction 方式
    """
    def __init__(self, losses: List[Tuple[str, nn.Module, float]] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights: Dict[str, float] = {}
        self.reduction = reduction

        if losses:
            for name, loss_fn, weight in losses:
                self.add_loss(name, loss_fn, weight)

    def add_loss(self, name: str, loss_fn: nn.Module, weight: float = 1.0):
        """添加一个损失函数"""
        self.losses[name] = loss_fn
        self.weights[name] = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        loss_details = {}

        for name in self.losses:
            loss_val = self.losses[name](inputs, targets)
            weighted = loss_val * self.weights[name]
            total_loss = total_loss + weighted
            loss_details[name] = loss_val.item()

        # 存储损失详情，供外部读取
        self._last_loss_details = loss_details

        if self.reduction == 'mean':
            total_weight = sum(self.weights.values())
            return total_loss / max(total_weight, 1e-8)
        return total_loss

    def get_loss_details(self) -> Dict[str, float]:
        """获取最近一次前向传播的各损失分量值"""
        return getattr(self, '_last_loss_details', {})


class TripletLoss(nn.Module):
    """Triplet Loss: 拉近同类样本，推远异类样本
    常用于度量学习、特征嵌入

    Args:
        margin: 正负样本对之间的最小距离
        p: 距离度量的范数 (1=L1, 2=L2)
    """
    def __init__(self, margin: float = 1.0, p: float = 2.0):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: torch.Tensor) -> torch.Tensor:
        d_ap = F.pairwise_distance(anchor, positive, p=self.p)
        d_an = F.pairwise_distance(anchor, negative, p=self.p)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive Loss: 对比损失
    用于孪生网络，拉近相似样本对，推远不相似样本对

    Args:
        margin: 不相似样本对的最小距离阈值
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """label: 1 表示相似，0 表示不相似"""
        euclidean_dist = F.pairwise_distance(output1, output2)
        loss_similar = label * euclidean_dist.pow(2)
        loss_dissimilar = (1 - label) * F.relu(self.margin - euclidean_dist).pow(2)
        return (loss_similar + loss_dissimilar).mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification
    非对称损失：对正负样本施加不同的聚焦力度
    适用于多标签分类场景

    Args:
        gamma_neg: 负样本聚焦参数
        gamma_pos: 正样本聚焦参数
        clip: 概率裁剪值
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 将 targets 转为 one-hot
        if targets.dim() == 1:
            targets = F.one_hot(targets, inputs.size(-1)).float()

        xs_pos = inputs
        xs_neg = 1 - inputs

        los_pos = targets * torch.log(xs_pos.clamp(min=self.clip))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.clip))

        loss = targets * los_pos + (1 - targets) * los_neg

        pt = targets * (1 - xs_pos) + (1 - targets) * xs_neg
        one_sided_gamma = targets * self.gamma_pos + (1 - targets) * self.gamma_neg
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)

        loss = loss * one_sided_w
        return -loss.sum() / max(targets.sum(), 1)


# ============================================================
# 损失函数注册表
# ============================================================

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "crossentropy": nn.CrossEntropyLoss,
    "nll": nn.NLLLoss,
    "nllloss": nn.NLLLoss,
    "bcewithlogits": nn.BCEWithLogitsLoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "bce": nn.BCELoss,
    "bceloss": nn.BCELoss,
    "mse": nn.MSELoss,
    "mseloss": nn.MSELoss,
    "l1": nn.L1Loss,
    "l1loss": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "smoothl1": nn.SmoothL1Loss,
    "huber": nn.SmoothL1Loss,
    "focal": FocalLoss,
    "focal_loss": FocalLoss,
    "label_smoothing": LabelSmoothingCrossEntropy,
    "label_smoothing_ce": LabelSmoothingCrossEntropy,
    "dice": DiceLoss,
    "dice_loss": DiceLoss,
    "asymmetric": AsymmetricLoss,
    "asl": AsymmetricLoss,
}


def create_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """损失函数工厂

    Args:
        loss_name: 损失函数名称（大小写不敏感）
        **kwargs: 损失函数参数，如:
            - focal_loss: gamma=2.0, alpha=0.25
            - label_smoothing: smoothing=0.1
            - dice_loss: smooth=1.0

    Returns:
        nn.Module 损失函数实例
    """
    key = loss_name.lower().strip().replace("_", "").replace("-", "")

    # 归一化映射
    mapping = {k.replace("_", "").replace("-", ""): v for k, v in LOSS_REGISTRY.items()}
    cls = mapping.get(key)

    if cls is None:
        raise ValueError(
            f"不支持的损失函数: {loss_name}，"
            f"支持: {', '.join(sorted(set(v.__name__ for v in LOSS_REGISTRY.values())))}"
        )

    # 标准 PyTorch 损失函数不需要额外参数
    if cls in (nn.CrossEntropyLoss, nn.NLLLoss, nn.BCEWithLogitsLoss,
               nn.BCELoss, nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss):
        return cls()

    # 自定义损失函数按需传参
    if cls == FocalLoss:
        return cls(alpha=kwargs.get("alpha", 0.25), gamma=kwargs.get("gamma", 2.0))
    if cls == LabelSmoothingCrossEntropy:
        return cls(smoothing=kwargs.get("smoothing", 0.1))
    if cls == DiceLoss:
        return cls(smooth=kwargs.get("smooth", 1.0))
    if cls == AsymmetricLoss:
        return cls(gamma_neg=kwargs.get("gamma_neg", 4.0),
                   gamma_pos=kwargs.get("gamma_pos", 1.0))

    return cls()


def list_loss_functions() -> List[str]:
    """列出所有支持的损失函数"""
    return sorted(set(v.__name__ for v in LOSS_REGISTRY.values()))