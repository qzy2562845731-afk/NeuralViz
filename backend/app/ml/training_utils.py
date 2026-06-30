"""
训练工具模块
- CheckpointManager: 断点续训 checkpoint 保存与恢复
- EarlyStopping: 早停机制
- LRSchedulerFactory: 学习率调度器工厂
- AMPManager: 混合精度训练管理
"""
import os
import json
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, ExponentialLR
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Checkpoint 管理
# ============================================================

class CheckpointManager:
    """训练断点续训管理器

    保存/恢复完整训练状态：模型权重、优化器状态、调度器状态、epoch、最佳指标
    """

    CHECKPOINT_FILENAME = "checkpoint.pt"

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.checkpoint_path = os.path.join(model_dir, self.CHECKPOINT_FILENAME)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        best_val_acc: float,
        best_epoch: int,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存完整训练检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch (1-based)
            best_val_acc: 最佳验证准确率
            best_epoch: 最佳 epoch
            scaler: AMP GradScaler（可选）
            scheduler: 学习率调度器（可选）
            extra: 额外元数据

        Returns:
            检查点文件路径
        """
        checkpoint = {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            checkpoint["extra"] = extra

        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint 已保存: epoch={epoch}, best_val_acc={best_val_acc:.4f}, path={self.checkpoint_path}")
        return self.checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scaler: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """加载检查点

        Returns:
            恢复的训练状态字典，若文件不存在返回 None
        """
        if not os.path.exists(self.checkpoint_path):
            return None

        checkpoint = torch.load(self.checkpoint_path, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(
            f"Checkpoint 已恢复: epoch={checkpoint.get('epoch', 0)}, "
            f"best_val_acc={checkpoint.get('best_val_acc', 0.0):.4f}"
        )
        return checkpoint

    def exists(self) -> bool:
        return os.path.exists(self.checkpoint_path)

    def remove(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)


# ============================================================
# 早停机制
# ============================================================

class EarlyStopping:
    """早停机制：监控验证集指标，在连续 patience 个 epoch 无改善时停止训练

    Args:
        patience: 容忍无改善的 epoch 数
        min_delta: 最小改善阈值
        mode: 'min' 监控 loss 下降 / 'max' 监控 accuracy 上升
        verbose: 是否打印日志
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score: float, epoch: int = 0) -> bool:
        """检查是否应该早停

        Args:
            current_score: 当前监控指标值
            epoch: 当前 epoch

        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience} 个 epoch 无改善 "
                    f"(best={self.best_score:.4f} at epoch {self.best_epoch})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


# ============================================================
# 学习率调度器工厂
# ============================================================

_SUPPORTED_SCHEDULERS = {
    "cosine": "CosineAnnealingLR",
    "cosine_annealing": "CosineAnnealingLR",
    "plateau": "ReduceLROnPlateau",
    "reduce_on_plateau": "ReduceLROnPlateau",
    "step": "StepLR",
    "exponential": "ExponentialLR",
    "none": None,
}


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    scheduler_params: Optional[Dict[str, Any]] = None,
    total_epochs: int = 20,
) -> Tuple[Optional[Any], str]:
    """创建学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 (cosine/plateau/step/exponential/none)
        scheduler_params: 调度器参数
        total_epochs: 总训练轮次

    Returns:
        (scheduler, scheduler_type_normalized)
    """
    params = scheduler_params or {}
    scheduler_type = scheduler_type.lower().strip().replace("-", "_")

    if scheduler_type == "none" or scheduler_type == "":
        return None, "none"

    if scheduler_type not in _SUPPORTED_SCHEDULERS:
        logger.warning(f"不支持的调度器类型: {scheduler_type}，将不使用调度器")
        return None, "none"

    if scheduler_type in ("cosine", "cosine_annealing"):
        t_max = params.get("t_max", total_epochs)
        eta_min = params.get("eta_min", params.get("min_lr", 0))
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return scheduler, "cosine_annealing"

    elif scheduler_type in ("plateau", "reduce_on_plateau"):
        mode = params.get("mode", "max")
        factor = params.get("factor", 0.5)
        patience = params.get("patience", 3)
        min_lr = params.get("min_lr", 1e-6)
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
        )
        return scheduler, "reduce_on_plateau"

    elif scheduler_type == "step":
        step_size = params.get("step_size", 10)
        gamma = params.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler, "step"

    elif scheduler_type == "exponential":
        gamma = params.get("gamma", 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        return scheduler, "exponential"

    return None, "none"


# ============================================================
# 混合精度训练 (AMP)
# ============================================================

class AMPManager:
    """自动混合精度训练管理器

    基于 torch.cuda.amp 实现，CPU 环境下自动降级为普通训练
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled) if self.enabled else None
        self.device_type = "cuda" if self.enabled else "cpu"

    def autocast_context(self):
        """返回 autocast 上下文管理器"""
        if self.enabled:
            return torch.cuda.amp.autocast()
        else:
            # CPU 下返回一个空的上下文管理器
            from contextlib import nullcontext
            return nullcontext()

    def backward(self, loss: torch.Tensor):
        """AMP-aware backward"""
        if self.enabled and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, optimizer: Optimizer):
        """AMP-aware optimizer step"""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self) -> Optional[Dict]:
        if self.scaler is not None:
            return self.scaler.state_dict()
        return None

    def load_state_dict(self, state_dict: Optional[Dict]):
        if self.scaler is not None and state_dict is not None:
            self.scaler.load_state_dict(state_dict)