"""
训练引擎服务层
异步后台训练、实时指标上报、自动模型归档
单例模式，统一管理所有训练任务
"""
import os
import json
import time
import random
import logging
import threading
import traceback
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

from app.core.security import sanitize_path_id
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.database import SessionLocal
from app.core.config import settings
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric
from app.services.dataset_service import DatasetService
from app.ml.model_builder import build_model, count_parameters, ConfigurableCNN, ConfigurableMLP
from app.ml.attention import SEBlock, CBAMBlock, SelfAttention2d
from app.ml.losses import create_loss_function, list_loss_functions
from app.ml.augmentation import (
    get_augmentation_transform, apply_augmentation_batch,
    get_batch_augmentation_config, cutmix_data, mixup_data,
)
from app.ml.metrics import compute_roc_curve, compute_pr_curve
from app.ml.training_utils import (
    CheckpointManager, EarlyStopping, create_scheduler, AMPManager,
)

logger = logging.getLogger(__name__)

# 默认超参数
_DEFAULT_HYPERPARAMS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "optimizer": "adam",
    "epochs": 20,
    "random_seed": 42,
    "val_split": 0.2,
    "loss_function": "cross_entropy",
    # 学习率调度
    "scheduler_type": "none",  # none / cosine / plateau / step / exponential
    "scheduler_params": {},     # 调度器参数字典
    # 早停
    "early_stopping": False,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
    "early_stopping_monitor": "val_accuracy",  # val_accuracy / val_loss
    # 混合精度
    "use_amp": False,  # 启用自动混合精度训练
    # 断点续训
    "resume_from_checkpoint": False,  # 从 checkpoint 恢复训练
}

# 支持的优化器清单
_SUPPORTED_OPTIMIZERS = {"adam", "sgd", "adamw", "rmsprop"}


class TrainingTask:
    """单个训练任务的运行时状态"""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.status: str = "pending"  # pending / running / completed / failed / stopped
        self.current_epoch: int = 0
        self.total_epochs: int = 0
        self.latest_metrics: Dict[str, Any] = {}
        self.logs: List[str] = []
        self.stop_flag: threading.Event = threading.Event()
        self.start_time: Optional[float] = None
        self.error: Optional[str] = None

    def log(self, message: str):
        """追加一条日志"""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        logger.info(f"[训练 {self.experiment_id}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "latest_metrics": self.latest_metrics,
            "error": self.error,
            "elapsed_seconds": round(time.time() - self.start_time, 1) if self.start_time else 0,
        }


class TrainingService:
    """训练引擎服务（单例）"""

    _instance: Optional['TrainingService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._tasks: Dict[str, TrainingTask] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="training")
        self._dataset_service = DatasetService()

    # ============================================================
    # 公共接口
    # ============================================================

    def start_training(
        self,
        experiment_id: str,
        dataset_id: Optional[str] = None,
        hyperparams_override: Optional[Dict[str, Any]] = None,
        model_config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """启动训练任务

        Args:
            experiment_id: 实验 ID
            dataset_id: 数据集 ID（可选，覆盖实验配置中的数据集）
            hyperparams_override: 超参数覆盖（可选）
            model_config_override: 模型结构配置覆盖（可选）

        Returns:
            任务状态字典
        """
        with self._lock:
            # 检查是否已有任务在运行
            existing = self._tasks.get(experiment_id)
            if existing and existing.status in ("pending", "running"):
                raise ValueError(f"实验 {experiment_id} 已有训练任务正在运行")

        # 校验实验
        exp_info = self._get_experiment(experiment_id)
        if not exp_info:
            raise ValueError("实验不存在")

        # 幂等校验：实验表 status 为 running 时也禁止重复启动（防止服务重启后内存字典清空导致重复提交）
        if exp_info.get("status") == "running":
            # 检查内存中是否真的有任务在跑
            with self._lock:
                existing = self._tasks.get(experiment_id)
                if existing and existing.status in ("pending", "running"):
                    raise ValueError(f"实验 {experiment_id} 正在训练中，请先停止当前训练")
            # 内存中无任务但数据库 status=running，说明服务重启过，允许重新启动
            logger.warning(f"[训练 {experiment_id}] 数据库状态为 running 但内存无任务，允许重新启动")

        # 解析数据集 ID
        # 深拷贝 config，避免污染 ExperimentService 单例缓存
        import copy
        config = copy.deepcopy(exp_info.get("config", {}) or {})
        if dataset_id:
            config["dataset_id"] = dataset_id
        else:
            dataset_id = config.get("dataset_id")
        if not dataset_id:
            raise ValueError("未指定数据集，请在启动训练时传入 dataset_id 或在实验 config 中配置")

        # 合并模型结构配置覆盖
        if model_config_override:
            config["model_config"] = copy.deepcopy(model_config_override)

        # 解析超参数
        hp = self._parse_hyperparams(exp_info, hyperparams_override)

        # 创建任务
        task = TrainingTask(experiment_id)
        task.total_epochs = hp["epochs"]
        with self._lock:
            self._tasks[experiment_id] = task

        # 根因修复：清理同一实验的旧指标，避免新旧数据拼接导致曲线断崖
        self._clear_old_metrics(experiment_id)

        # 更新实验状态为 running
        self._update_experiment(experiment_id, status="running", total_epochs=hp["epochs"])

        # 提交到线程池
        self._executor.submit(self._run_training, experiment_id, dataset_id, hp, config)

        return task.to_dict()

    def stop_training(self, experiment_id: str) -> Dict[str, Any]:
        """停止训练任务（优雅退出）"""
        with self._lock:
            task = self._tasks.get(experiment_id)
            if not task:
                raise ValueError(f"实验 {experiment_id} 无训练任务")
            if task.status not in ("pending", "running"):
                raise ValueError(f"实验 {experiment_id} 训练任务已结束（{task.status}）")
            task.stop_flag.set()
            task.log("收到停止信号，将在当前 epoch 结束后退出")
        return self.get_status(experiment_id)

    def get_status(self, experiment_id: str) -> Dict[str, Any]:
        """获取训练实时状态"""
        with self._lock:
            task = self._tasks.get(experiment_id)
            if not task:
                return {
                    "experiment_id": experiment_id,
                    "status": "idle",
                    "message": "无训练任务",
                }
            return task.to_dict()

    def get_logs(self, experiment_id: str, since: int = 0) -> Dict[str, Any]:
        """获取训练日志（增量）"""
        with self._lock:
            task = self._tasks.get(experiment_id)
            if not task:
                return {
                    "experiment_id": experiment_id,
                    "logs": [],
                    "total": 0,
                }
            logs = task.logs[since:]
            return {
                "experiment_id": experiment_id,
                "logs": logs,
                "total": len(task.logs),
            }

    def get_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """获取全量训练指标时序数据"""
        from app.services.experiment_service import ExperimentService
        exp_service = ExperimentService()
        metrics = exp_service.get_metrics(experiment_id, limit=10000)
        return {
            "experiment_id": experiment_id,
            "metrics": metrics,
            "count": len(metrics),
        }

    # ============================================================
    # 训练核心流程
    # ============================================================

    def _run_training(
        self,
        experiment_id: str,
        dataset_id: str,
        hp: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """训练主流程（在线程池中执行）"""
        experiment_id = sanitize_path_id(experiment_id)
        task = self._tasks[experiment_id]
        task.status = "running"
        task.start_time = time.time()

        try:
            # 1. 设置随机种子
            self._set_seed(hp["random_seed"])
            task.log(f"随机种子: {hp['random_seed']}")

            # 2. 加载数据集
            task.log(f"加载数据集: {dataset_id}")
            ds_data = self._dataset_service.load_dataset(dataset_id)
            X, y = ds_data["X"], ds_data["y"]
            dataset_type = ds_data["dataset_type"]
            feature_shape = ds_data["feature_shape"]
            # 校验数据集有效性
            if len(y) == 0:
                raise ValueError("数据集为空，无法训练")
            if not feature_shape:
                raise ValueError("数据集 feature_shape 为空，无法构建模型")
            task.log(f"数据集加载完成: {len(y)} 样本, 类型={dataset_type}, 维度={feature_shape}")

            # 3. 数据预处理
            X_tensor, y_tensor, num_classes = self._prepare_data(X, y, dataset_type, feature_shape)
            task.log(f"数据预处理完成: 输入形状={list(X_tensor.shape)}, 类别数={num_classes}")

            # 4. 划分训练集/验证集
            X_train, y_train, X_val, y_val = self._split_data(X_tensor, y_tensor, hp["val_split"])
            task.log(f"数据划分: 训练集={len(y_train)}, 验证集={len(y_val)}")

            # 5. 构建模型（支持 model_config：注意力机制、通道数、消融开关等）
            model_config = self._extract_model_config(config, hp)
            model = build_model(dataset_type, feature_shape, num_classes, model_config)
            total_params = count_parameters(model)
            attention_type = model_config.get("attention", "none") if dataset_type in ("image_folder", "mnist_idx") else ("se" if model_config.get("use_attention") else "none")

            # 5.1 数据增强配置
            augmentation_config = config.get("augmentation", {}) or {}
            use_augmentation = augmentation_config.get("enabled", False) and dataset_type in ("image_folder", "mnist_idx")
            train_transform = get_augmentation_transform(augmentation_config, is_train=True) if use_augmentation else None
            if use_augmentation:
                task.log(f"数据增强已启用: {augmentation_config}")

            # 提取 PyTorch 模型层信息（用于实验记录中的模型结构展示）
            pytorch_layers = []
            for name, module in model.named_modules():
                if name == "":
                    continue
                # 跳过容器类模块，只记录叶子层
                if len(list(module.children())) > 0:
                    continue
                layer_info = {
                    "name": name,
                    "type": module.__class__.__name__,
                }
                # 提取常见层参数
                if hasattr(module, "in_channels"):
                    layer_info["in_channels"] = module.in_channels
                if hasattr(module, "out_channels"):
                    layer_info["out_channels"] = module.out_channels
                if hasattr(module, "in_features"):
                    layer_info["in_features"] = module.in_features
                if hasattr(module, "out_features"):
                    layer_info["out_features"] = module.out_features
                if hasattr(module, "kernel_size"):
                    ks = module.kernel_size
                    layer_info["kernel_size"] = list(ks) if isinstance(ks, tuple) else ks
                if hasattr(module, "stride"):
                    sd = module.stride
                    layer_info["stride"] = list(sd) if isinstance(sd, tuple) else sd
                if hasattr(module, "padding"):
                    pd = module.padding
                    layer_info["padding"] = list(pd) if isinstance(pd, tuple) else pd
                if hasattr(module, "p"):
                    layer_info["dropout"] = module.p
                # 计算该层参数量
                layer_params = sum(p.numel() for p in module.parameters(recurse=False))
                layer_info["params"] = layer_params
                pytorch_layers.append(layer_info)

            # 构建完整的 model_architecture JSON
            final_model_arch = {
                "type": dataset_type,
                "name": model.__class__.__name__,
                "class_name": model.__class__.__name__,
                "total_params": total_params,
                "num_layers": len(pytorch_layers),
                "layers": pytorch_layers,
                "input_shape": list(feature_shape) if isinstance(feature_shape, (list, tuple)) else feature_shape,
                "num_classes": num_classes,
            }

            task.log(f"模型构建完成: {model.__class__.__name__}, 参数量={total_params:,}, 层数={len(pytorch_layers)}")
            if dataset_type in ("image_folder", "mnist_idx"):
                ch_info = "→".join(str(c) for c in model_config.get("channels", [32, 64]))
                attn_info = attention_type if attention_type != "none" else "无"
                task.log(f"卷积通道配置: {ch_info}, 注意力机制: {attn_info}, "
                         f"BN={model_config.get('use_bn', True)}, Dropout={model_config.get('use_dropout', True)}, "
                         f"残差={model_config.get('use_residual', False)}")

            # 6. 配置优化器、损失函数、AMP、调度器、早停
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = self._build_optimizer(model, hp)
            criterion = self._build_loss_function(hp)
            loss_display_name = hp.get("loss_function", "cross_entropy").title().replace("_", "")

            # 混合精度训练
            amp_manager = AMPManager(enabled=hp["use_amp"]) if hp["use_amp"] else None
            if amp_manager is not None and amp_manager.enabled:
                task.log(f"混合精度训练(AMP)已启用，设备: {amp_manager.device_type}")
            else:
                task.log(f"设备: {device}")

            # 学习率调度器
            scheduler, scheduler_type = create_scheduler(
                optimizer,
                hp["scheduler_type"],
                hp["scheduler_params"],
                total_epochs=hp["epochs"],
            )
            if scheduler is not None:
                task.log(f"学习率调度器: {scheduler_type}")

            # 早停机制
            early_stopping = None
            if hp["early_stopping"]:
                monitor = hp["early_stopping_monitor"]
                mode = "max" if monitor == "val_accuracy" else "min"
                early_stopping = EarlyStopping(
                    patience=hp["early_stopping_patience"],
                    min_delta=hp["early_stopping_min_delta"],
                    mode=mode,
                )
                task.log(f"早停机制: 监控 {monitor}, patience={hp['early_stopping_patience']}")

            # Batch-level 增强配置
            augmentation_config = config.get("augmentation", {}) or {}
            batch_aug = get_batch_augmentation_config(augmentation_config) if augmentation_config else None
            if batch_aug and batch_aug.get("use_cutmix"):
                task.log(f"CutMix 已启用 (alpha={batch_aug['cutmix_alpha']})")
            if batch_aug and batch_aug.get("use_mixup"):
                task.log(f"MixUp 已启用 (alpha={batch_aug['mixup_alpha']})")

            # 训练日志首行
            task.log(f"=== 训练启动 ===")
            task.log(f"损失函数: {loss_display_name}, 优化器: {hp['optimizer']}(lr={hp['learning_rate']})")
            task.log(f"batch_size={hp['batch_size']}, epochs={hp['epochs']}, random_seed={hp['random_seed']}, val_split={hp['val_split']}")

            # 7. 构建 DataLoader
            train_loader = self._make_loader(X_train, y_train, hp["batch_size"], shuffle=True)
            val_loader = self._make_loader(X_val, y_val, hp["batch_size"], shuffle=False)

            # 8. 模型目录 & checkpoint 管理
            model_dir = settings.MODEL_DIR / experiment_id
            model_dir.mkdir(parents=True, exist_ok=True)
            best_path = str(model_dir / "best.pt")
            last_path = str(model_dir / "last.pt")
            checkpoint_mgr = CheckpointManager(str(model_dir))

            # 初始化训练状态
            best_val_acc = 0.0
            best_epoch = 0
            start_epoch = 1
            train_loss = val_loss = 0.0
            train_acc = val_acc = 0.0
            precision = recall = f1 = 0.0
            grad_norm = 0.0
            confusion_matrix = None

            # 断点续训：尝试加载 checkpoint
            if hp["resume_from_checkpoint"] and checkpoint_mgr.exists():
                ckpt = checkpoint_mgr.load(
                    model, optimizer,
                    scaler=amp_manager.scaler if amp_manager else None,
                    scheduler=scheduler,
                )
                if ckpt:
                    start_epoch = ckpt.get("epoch", 0) + 1
                    best_val_acc = ckpt.get("best_val_acc", 0.0)
                    best_epoch = ckpt.get("best_epoch", 0)
                    task.log(f"从 checkpoint 恢复训练: epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")
                    # 重新加载旧指标用于继续追踪
                    task.log(f"跳过已完成的 {start_epoch - 1} 个 epoch")

            # 9. 训练循环
            for epoch in range(start_epoch, hp["epochs"] + 1):
                # 检查停止信号
                if task.stop_flag.is_set():
                    task.status = "stopped"
                    task.log(f"训练在第 {epoch-1} epoch 后被用户停止")
                    break

                task.current_epoch = epoch
                current_lr = optimizer.param_groups[0]["lr"]
                task.log(f"--- Epoch {epoch}/{hp['epochs']} (lr={current_lr:.6f}) ---")

                # 训练阶段
                train_loss, train_acc, grad_norm = self._train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    transform=train_transform if use_augmentation else None,
                    dataset_type=dataset_type,
                    amp_manager=amp_manager,
                    batch_aug=batch_aug,
                    num_classes=num_classes,
                )

                # 验证阶段
                val_loss, val_acc, precision, recall, f1, confusion_matrix, all_preds, all_labels, all_probs, per_class_p, per_class_r, per_class_f1 = self._validate(
                    model, val_loader, criterion, device, num_classes
                )

                # 学习率调度器更新
                if scheduler is not None:
                    if scheduler_type == "reduce_on_plateau":
                        scheduler.step(val_acc if hp["early_stopping_monitor"] == "val_accuracy" else val_loss)
                    else:
                        scheduler.step()

                # 权重范数
                weight_norm = 0.0
                for p in model.parameters():
                    weight_norm += p.data.norm(2).item() ** 2
                weight_norm = weight_norm ** 0.5

                # 可视化数据
                vis_data = self._extract_visualization_data(
                    model, X_val, all_preds, num_classes
                )

                # ROC/PR 曲线（最后一个epoch）
                roc_data = None
                pr_data = None
                is_last_epoch = (epoch == hp["epochs"]) and not task.stop_flag.is_set()
                if is_last_epoch and len(all_probs) > 0 and num_classes > 1:
                    try:
                        ds_info = self._dataset_service.get_dataset(dataset_id)
                        class_names = None
                        if ds_info:
                            cd = ds_info.get("class_distribution", {})
                            if cd:
                                class_names = list(cd.keys())[:num_classes]
                        roc_data = compute_roc_curve(all_labels, all_probs, num_classes, class_names)
                        pr_data = compute_pr_curve(all_labels, all_probs, num_classes, class_names)
                        task.log(f"ROC AUC (macro): {roc_data['macro_auc']:.4f}, PR AP (macro): {pr_data['macro_ap']:.4f}")
                    except Exception as e:
                        logger.warning(f"计算ROC/PR指标失败: {e}")

                # 记录指标
                metrics = {
                    "epoch": epoch,
                    "step": epoch,
                    "loss": round(train_loss, 6),
                    "accuracy": round(train_acc, 4),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 4),
                    "learning_rate": current_lr,
                    "batch_size": hp["batch_size"],
                    "metric_type": "training",
                    "extra_data": json.dumps({
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "gradient_norm": round(grad_norm, 4),
                        "weight_norm": round(weight_norm, 4),
                        "confusion_matrix": confusion_matrix,
                        "per_class_precision": per_class_p,
                        "per_class_recall": per_class_r,
                        "per_class_f1": per_class_f1,
                        "layer_activations": vis_data["layer_activations"],
                        "feature_maps": vis_data["feature_maps"],
                        "prediction_distribution": vis_data["prediction_distribution"],
                        "roc_curve": roc_data,
                        "pr_curve": pr_data,
                    }),
                }
                task.latest_metrics = metrics
                self._save_metric(experiment_id, metrics)

                task.log(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, f1={f1:.4f}"
                )

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_path)
                    task.log(f"  -> 新最佳模型已保存 (val_acc={val_acc:.4f})")

                # 保存 checkpoint（每个 epoch 都保存，支持断点续训）
                checkpoint_mgr.save(
                    model, optimizer, epoch, best_val_acc, best_epoch,
                    scaler=amp_manager.scaler if amp_manager else None,
                    scheduler=scheduler,
                )

                # 更新实验进度
                self._update_experiment(
                    experiment_id,
                    current_step=epoch,
                    best_accuracy=best_val_acc,
                )

                # 早停检查
                if early_stopping is not None:
                    monitor_value = val_acc if hp["early_stopping_monitor"] == "val_accuracy" else val_loss
                    if early_stopping(monitor_value, epoch):
                        task.log(f"早停触发! 最佳 {hp['early_stopping_monitor']}={early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
                        task.status = "completed"
                        break

            # 9. 保存最终模型
            torch.save(model.state_dict(), last_path)
            task.log(f"最终模型已保存: {last_path}")

            # 9.5 保存特征图与卷积核可视化数据（仅图像模型）
            visualization_paths = {}
            if dataset_type in ("image_folder", "mnist_idx") and isinstance(model, ConfigurableCNN):
                try:
                    visualization_paths = self._save_cnn_visualizations(
                        model, X_val, model_dir, device
                    )
                    if visualization_paths:
                        task.log(f"CNN可视化数据已保存: {list(visualization_paths.keys())}")
                except Exception as e:
                    logger.warning(f"保存CNN可视化数据失败（不影响训练）: {e}")

            # 10. 训练完成标记（必须在构建config之前记录日志，确保日志完整）
            if not task.stop_flag.is_set():
                task.status = "completed"

            # 11. 计算训练耗时
            training_duration = time.time() - task.start_time
            final_loss = train_loss

            if task.status == "completed":
                task.log(f"训练完成！最佳 val_acc={best_val_acc:.4f} (epoch {best_epoch}), 耗时 {training_duration:.1f}s")

            # 12. 构建最终config汇总（包含完整日志）
            config.update({
                "dataset_id": dataset_id,
                "dataset_name": ds_data.get("dataset_name", "") or config.get("dataset_name", ""),
                "loss_function": loss_display_name,
                "best_epoch": best_epoch,
                "best_val_accuracy": round(best_val_acc, 4),
                "final_train_loss": round(final_loss, 6),
                "final_val_loss": round(val_loss, 6),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "training_duration": round(training_duration, 2),
                "model_path_best": best_path,
                "model_path_last": last_path,
                "model_architecture": model.__class__.__name__,
                "model_config": model_config,
                "attention_type": attention_type,
                "total_params": total_params,
                "num_classes": num_classes,
                "training_logs": task.logs,
                "visualizations": visualization_paths,
                # 新增训练策略配置
                "scheduler_type": scheduler_type,
                "use_amp": hp["use_amp"],
                "early_stopping": hp["early_stopping"],
                "augmentation": augmentation_config,
                "checkpoint_path": checkpoint_mgr.checkpoint_path if checkpoint_mgr.exists() else None,
            })

            # 回写实际生效的超参数到 hyperparams 字段
            final_hyperparams = {
                "learning_rate": hp["learning_rate"],
                "batch_size": hp["batch_size"],
                "optimizer": hp["optimizer"],
                "epochs": hp["epochs"],
                "random_seed": hp["random_seed"],
                "val_split": hp["val_split"],
                "loss_function": loss_display_name,
                "scheduler_type": scheduler_type,
                "use_amp": hp["use_amp"],
                "early_stopping": hp["early_stopping"],
                "early_stopping_patience": hp["early_stopping_patience"],
                "early_stopping_monitor": hp["early_stopping_monitor"],
            }

            self._update_experiment(
                experiment_id,
                status=task.status,
                best_accuracy=best_val_acc,
                final_loss=final_loss,
                total_epochs=task.current_epoch,
                config=config,
                hyperparams=final_hyperparams,
                model_architecture=final_model_arch,
                total_params=total_params,
                layer_count=len(pytorch_layers),
            )

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            error_stack = traceback.format_exc()
            task.log(f"训练失败: {e}")
            task.log(error_stack)
            logger.error(f"[训练 {experiment_id}] 训练失败", exc_info=True)
            # be14/be16修复：异常终止时也回填已产生的数据，不丢失
            # 使用 locals().get() 防止异常发生在变量初始化前导致 NameError
            _best_epoch = locals().get("best_epoch", 0)
            _best_val_acc = locals().get("best_val_acc", 0.0)
            _final_loss = locals().get("final_loss", locals().get("train_loss", 0.0))
            try:
                partial_duration = time.time() - task.start_time
                config.update({
                    "best_epoch": _best_epoch,
                    "training_duration": round(partial_duration, 2),
                    "training_logs": task.logs,
                    "error_msg": str(e),
                    "error_stack": error_stack,
                    "loss_function": locals().get("loss_display_name", "CrossEntropyLoss"),
                })
                # 异常时也回写 hyperparams
                _hp = locals().get("hp", {})
                _final_hp = {
                    "learning_rate": _hp.get("learning_rate", 0.001),
                    "batch_size": _hp.get("batch_size", 32),
                    "optimizer": _hp.get("optimizer", "adam"),
                    "epochs": _hp.get("epochs", 20),
                    "random_seed": _hp.get("random_seed", 42),
                    "val_split": _hp.get("val_split", 0.2),
                    "loss_function": locals().get("loss_display_name", "CrossEntropyLoss"),
                } if _hp else None
                _update_kwargs = dict(
                    status="failed",
                    best_accuracy=_best_val_acc,
                    final_loss=_final_loss,
                    total_epochs=task.current_epoch,
                    config=config,
                )
                if _final_hp:
                    _update_kwargs["hyperparams"] = _final_hp
                self._update_experiment(experiment_id, **_update_kwargs)
            except Exception as update_err:
                logger.error(f"[训练 {experiment_id}] 回填失败数据时出错: {update_err}")
        finally:
            # 安全兜底：确保终态状态与汇总字段始终同步到数据库
            # 防止 _update_experiment 在正常路径中失败导致状态卡在 running、config 字段丢失
            if task.status not in ("pending", "running"):
                try:
                    from app.services.experiment_service import ExperimentService
                    exp_service = ExperimentService()
                    current = exp_service.get_experiment(experiment_id)
                    if current and current.get("status") not in ("completed", "failed", "stopped"):
                        # 补全 config 中的汇总字段，确保 best_epoch/training_duration/training_logs 不丢失
                        fallback_update = {"status": task.status}
                        try:
                            _best_epoch = locals().get("best_epoch", 0)
                            _best_val_acc = locals().get("best_val_acc", 0.0)
                            _final_loss = locals().get("final_loss", locals().get("train_loss", 0.0))
                            partial_duration = time.time() - task.start_time
                            config.update({
                                "best_epoch": _best_epoch,
                                "training_duration": round(partial_duration, 2),
                                "training_logs": task.logs,
                            })
                            fallback_update["config"] = config
                            fallback_update["best_accuracy"] = _best_val_acc
                            fallback_update["final_loss"] = _final_loss
                            fallback_update["total_epochs"] = task.current_epoch
                        except Exception as cfg_err:
                            logger.warning(f"[训练 {experiment_id}] finally兜底补全config失败，仅更新status: {cfg_err}")
                        exp_service.update_experiment(experiment_id, **fallback_update)
                        logger.info(f"[训练 {experiment_id}] finally兜底：状态已同步为 {task.status}")
                except Exception as e:
                    logger.error(f"[训练 {experiment_id}] finally兜底更新状态失败: {e}")

    # ============================================================
    # 模型配置解析
    # ============================================================

    @staticmethod
    def _extract_model_config(config: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
        """从实验config中提取模型配置，用于构建ConfigurableCNN/ConfigurableMLP

        支持的配置字段（放在 config.model_config 或直接在 config 根级别）：
        - channels: List[int]  卷积通道数，默认 [32,64]
        - attention: str       注意力类型 none/se/eca/cbam/self_attention/mhsa/gct/coord
        - activation: str      激活函数 relu/leaky_relu/gelu/silu/tanh
        - attention_kwargs: Dict 注意力额外参数
        - use_bn: bool         是否使用批归一化
        - use_dropout: bool    是否使用Dropout
        - dropout_rate: float  Dropout比例
        - use_residual: bool   是否使用残差连接
        - fc_hidden: int       FC层隐藏维度
        - use_attention: bool  MLP是否使用SE-like特征重标定
        """
        mc = config.get("model_config", {}) or {}
        # 也支持从 config 根级别读取（消融实验字段）
        channels = mc.get("channel_list") or mc.get("channels") or config.get("channels") or [32, 64]
        model_config = {
            "channels": channels,
            "attention": mc.get("attention", config.get("attention", "none")),
            "activation": mc.get("activation", config.get("activation", "relu")),
            "attention_kwargs": mc.get("attention_kwargs", config.get("attention_kwargs")),
            "use_bn": mc.get("use_bn", config.get("use_bn", True)),
            "use_dropout": mc.get("use_dropout", config.get("use_dropout", True)),
            "dropout_rate": mc.get("dropout_rate", config.get("dropout_rate", 0.3)),
            "use_residual": mc.get("use_residual", config.get("use_residual", False)),
            "fc_hidden": mc.get("fc_hidden", config.get("fc_hidden", 128)),
            "use_attention": mc.get("use_attention", config.get("use_attention", False)),
        }
        # 确保 channels 是 list[int]
        if not isinstance(model_config["channels"], (list, tuple)):
            model_config["channels"] = [32, 64]
        model_config["channels"] = [int(c) for c in model_config["channels"]]
        # 类型校验
        model_config["use_bn"] = bool(model_config["use_bn"])
        model_config["use_dropout"] = bool(model_config["use_dropout"])
        model_config["use_residual"] = bool(model_config["use_residual"])
        model_config["use_attention"] = bool(model_config["use_attention"])
        model_config["dropout_rate"] = float(model_config["dropout_rate"])
        model_config["fc_hidden"] = int(model_config["fc_hidden"])
        return model_config

    # ============================================================
    # CNN 可视化保存（特征图 + 卷积核）
    # ============================================================

    def _save_cnn_visualizations(
        self,
        model: "ConfigurableCNN",
        X_val: torch.Tensor,
        model_dir,
        device: torch.device,
    ) -> Dict[str, Any]:
        """训练完成后保存特征图和卷积核可视化数据到JSON

        Returns:
            dict with keys: feature_maps_file, kernels_file, num_conv_layers
        """
        import json as _json
        model.eval()
        vis_dir = model_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        result = {}

        # 1. 特征图：取验证集第一个样本，提取每一层的特征图
        with torch.no_grad():
            sample = X_val[:1].to(device)
            try:
                feature_maps = model.get_feature_maps(sample)
            except Exception:
                feature_maps = {}

        fm_data = {}
        for layer_name, fm_tensor in feature_maps.items():
            # fm_tensor shape: (1, C, H, W)
            fm_np = fm_tensor[0].cpu().numpy()  # (C, H, W)
            num_ch = min(8, fm_np.shape[0])  # 每层最多取8个通道
            channels_data = []
            for c in range(num_ch):
                ch = fm_np[c]
                # 归一化到 [0, 1]
                cmin, cmax = ch.min(), ch.max()
                if cmax > cmin:
                    ch = (ch - cmin) / (cmax - cmin)
                # 降采样到 16x16 以便存储
                import torch.nn.functional as F_t
                ch_t = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).float()
                ch_resized = F_t.adaptive_avg_pool2d(ch_t, (16, 16)).squeeze().numpy()
                channels_data.append([[round(float(v), 4) for v in row] for row in ch_resized])
            fm_data[layer_name] = {
                "shape": list(fm_np.shape),
                "num_channels_sampled": num_ch,
                "channels": channels_data,
            }

        fm_path = str(vis_dir / "feature_maps.json")
        with open(fm_path, "w", encoding="utf-8") as f:
            _json.dump(fm_data, f, ensure_ascii=False)
        result["feature_maps_file"] = fm_path
        result["num_conv_layers"] = len(fm_data)

        # 2. 卷积核权重：提取每一层卷积核
        kernels_data = {}
        for layer_name, kw_tensor in model.get_kernels().items():
            # kw_tensor shape: (out_ch, in_ch, kH, kW)
            kw_np = kw_tensor.cpu().numpy()
            out_ch = min(8, kw_np.shape[0])
            in_ch = min(4, kw_np.shape[1])
            kernels_data[layer_name] = {
                "shape": list(kw_np.shape),
                "out_channels_sampled": out_ch,
                "in_channels_sampled": in_ch,
                "kernels": [],
            }
            for oc in range(out_ch):
                oc_kernels = []
                for ic in range(in_ch):
                    k = kw_np[oc, ic]
                    kmin, kmax = k.min(), k.max()
                    if kmax > kmin:
                        k = (k - kmin) / (kmax - kmin)
                    oc_kernels.append([[round(float(v), 4) for v in row] for row in k])
                kernels_data[layer_name]["kernels"].append(oc_kernels)

        kw_path = str(vis_dir / "kernels.json")
        with open(kw_path, "w", encoding="utf-8") as f:
            _json.dump(kernels_data, f, ensure_ascii=False)
        result["kernels_file"] = kw_path

        # 3. 注意力热力图（如果有注意力模块）
        attn_data = {}
        for name, module in model.named_modules():
            if isinstance(module, (SEBlock, CBAMBlock)):
                # 用 sample 重新前向传播并捕获注意力权重
                try:
                    captured = {}
                    def make_hook(lbl):
                        def hook(mod, inp, out):
                            if isinstance(mod, SEBlock):
                                # SE 通过 Sigmoid 输出通道权重
                                with torch.no_grad():
                                    b, c, _, _ = inp[0].size()
                                    y = mod.avg_pool(inp[0]).view(b, c)
                                    w = mod.fc(y)
                                    captured[lbl] = w[0].cpu().numpy().tolist()
                        return hook
                    h = module.register_forward_hook(make_hook(name))
                    with torch.no_grad():
                        model(sample)
                    h.remove()
                    if captured:
                        attn_data[name] = captured
                except Exception:
                    pass

        if attn_data:
            attn_path = str(vis_dir / "attention_weights.json")
            with open(attn_path, "w", encoding="utf-8") as f:
                _json.dump(attn_data, f, ensure_ascii=False)
            result["attention_weights_file"] = attn_path

        return result

    # ============================================================
    # 训练辅助方法
    # ============================================================

    def _set_seed(self, seed: int):
        """固定随机种子，保证可复现"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _parse_hyperparams(
        self,
        exp_info: Dict[str, Any],
        override: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """合并默认超参数、实验配置与覆盖参数，并执行合法性校验"""
        hp = dict(_DEFAULT_HYPERPARAMS)
        exp_hp = exp_info.get("hyperparams", {}) or {}
        # 从实验 hyperparams 中读取
        for key in _DEFAULT_HYPERPARAMS:
            if key in exp_hp:
                hp[key] = exp_hp[key]
        # 覆盖参数
        if override:
            for key in _DEFAULT_HYPERPARAMS:
                if key in override:
                    hp[key] = override[key]
        # 类型转换
        hp["learning_rate"] = float(hp["learning_rate"])
        hp["batch_size"] = int(hp["batch_size"])
        hp["epochs"] = int(hp["epochs"])
        hp["random_seed"] = int(hp["random_seed"])
        hp["val_split"] = float(hp["val_split"])
        hp["optimizer"] = str(hp["optimizer"]).lower().strip()
        hp["loss_function"] = str(hp.get("loss_function", "cross_entropy")).lower().strip()
        hp["scheduler_type"] = str(hp.get("scheduler_type", "none")).lower().strip()
        hp["scheduler_params"] = hp.get("scheduler_params", {}) or {}
        hp["early_stopping"] = bool(hp.get("early_stopping", False))
        hp["early_stopping_patience"] = int(hp.get("early_stopping_patience", 5))
        hp["early_stopping_min_delta"] = float(hp.get("early_stopping_min_delta", 0.0))
        hp["early_stopping_monitor"] = str(hp.get("early_stopping_monitor", "val_accuracy")).lower().strip()
        hp["use_amp"] = bool(hp.get("use_amp", False))
        hp["resume_from_checkpoint"] = bool(hp.get("resume_from_checkpoint", False))

        # ===== 合法性校验：异常提前拦截 =====
        if not (1e-5 <= hp["learning_rate"] <= 1.0):
            raise ValueError(f"学习率必须在 0.00001 ~ 1.0 之间，当前值: {hp['learning_rate']}")
        if hp["batch_size"] < 1:
            raise ValueError(f"批次大小必须 >= 1，当前值: {hp['batch_size']}")
        if hp["epochs"] < 1:
            raise ValueError(f"训练轮次必须 >= 1，当前值: {hp['epochs']}")
        if not (0.0 < hp["val_split"] < 1.0):
            raise ValueError(f"验证集比例必须在 (0, 1) 之间，当前值: {hp['val_split']}")
        if hp["optimizer"] not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"不支持的优化器: {hp['optimizer']}，支持的优化器: {', '.join(sorted(_SUPPORTED_OPTIMIZERS))}"
            )
        # 损失函数校验：使用与 create_loss_function 相同的归一化逻辑
        norm_loss = hp["loss_function"].lower().replace("_", "").replace("-", "")
        from app.ml.losses import LOSS_REGISTRY
        if norm_loss not in {k.replace("_", "").replace("-", ""): True for k in LOSS_REGISTRY}:
            raise ValueError(
                f"不支持的损失函数: {hp['loss_function']}，支持的损失函数: {', '.join(sorted(set(LOSS_REGISTRY.keys())))}"
            )
        return hp

    def _build_loss_function(self, hp: Dict[str, Any]) -> nn.Module:
        """根据超参数构建损失函数，使用统一工厂函数"""
        loss_name = hp.get("loss_function", "cross_entropy")
        loss_kwargs = hp.get("loss_kwargs", {}) or {}
        return create_loss_function(loss_name, **loss_kwargs)

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_type: str,
        feature_shape: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """将 numpy 数据预处理为 PyTorch 张量

        - 图像数据：归一化到 [0,1]，转为 NCHW
        - 表格数据：转为 float32
        - 标签：转为 int64
        """
        # 标签处理
        y = np.array(y).astype(np.int64)
        # 标签可能不是从 0 开始的连续整数，做映射
        unique_labels = sorted(set(y.tolist()))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y], dtype=np.int64)
        num_classes = len(unique_labels)

        if dataset_type in ("image_folder", "mnist_idx"):
            # 图像数据预处理
            X = np.array(X, dtype=np.float32)
            # 归一化到 [0, 1]
            if X.max() > 1.0:
                X = X / 255.0

            # 确保是 4D: (N, C, H, W)
            if X.ndim == 2:
                # (N, 784) -> (N, 1, 28, 28)
                X = X.reshape(-1, 1, 28, 28)
            elif X.ndim == 3:
                # (N, H, W) -> (N, 1, H, W)
                X = X[:, np.newaxis, :, :]
            elif X.ndim == 4:
                # (N, H, W, C) -> (N, C, H, W)
                if X.shape[-1] in (1, 3, 4):
                    X = np.transpose(X, (0, 3, 1, 2))
            X_tensor = torch.from_numpy(X).float()
        else:
            # 表格数据
            X = np.array(X, dtype=np.float32)
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            X_tensor = torch.from_numpy(X).float()

        y_tensor = torch.from_numpy(y).long()
        return X_tensor, y_tensor, num_classes

    def _split_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_split: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """按比例划分训练集/验证集"""
        n = len(y)
        indices = torch.randperm(n)
        val_count = max(1, int(n * val_split)) if n > 1 else 0
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

        if val_count == 0:
            # 数据太少，全部作为训练集，验证集复用训练集
            return X, y, X, y

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def _make_loader(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """构建 DataLoader"""
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(y)),
            shuffle=shuffle,
            drop_last=False,
        )

    def _build_optimizer(self, model: nn.Module, hp: Dict[str, Any]) -> torch.optim.Optimizer:
        """构建优化器：支持 Adam/SGD/AdamW/RMSprop，SGD 默认 momentum=0.9"""
        lr = hp["learning_rate"]
        opt_name = hp["optimizer"].lower()
        if opt_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        elif opt_name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            return torch.optim.Adam(model.parameters(), lr=lr)

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        transform: Optional[Any] = None,
        dataset_type: str = "",
        amp_manager: Optional[AMPManager] = None,
        batch_aug: Optional[Dict[str, Any]] = None,
        num_classes: int = 10,
    ) -> Tuple[float, float, float]:
        """训练一个 epoch，返回 (平均loss, 准确率, 梯度范数)

        Args:
            model: 模型
            loader: 训练数据加载器
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            transform: 数据增强变换（可选）
            dataset_type: 数据集类型
            amp_manager: AMP 混合精度管理器
            batch_aug: batch-level 增强配置 (CutMix/MixUp)
            num_classes: 类别数
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norm_sum = 0.0
        num_batches = 0

        use_cutmix = batch_aug and batch_aug.get("use_cutmix", False)
        use_mixup = batch_aug and batch_aug.get("use_mixup", False)
        cutmix_alpha = (batch_aug or {}).get("cutmix_alpha", 1.0)
        mixup_alpha = (batch_aug or {}).get("mixup_alpha", 1.0)

        for X_batch, y_batch in loader:
            # 应用数据增强（仅对图像训练集）
            if transform is not None and dataset_type in ("image_folder", "mnist_idx"):
                try:
                    X_np = X_batch.cpu().numpy()
                    X_np = apply_augmentation_batch(X_np, transform)
                    X_batch = torch.from_numpy(X_np).float()
                except Exception as e:
                    logger.warning(f"数据增强应用失败，使用原始数据: {e}")

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_np = y_batch.cpu().numpy()

            # Batch-level 增强：CutMix / MixUp
            use_batch_aug = False
            if use_cutmix and device.type == "cpu":  # CutMix 在训练循环中应用
                X_np = X_batch.cpu().numpy()
                X_np, y_mixed = cutmix_data(X_np, y_batch_np, alpha=cutmix_alpha, num_classes=num_classes)
                X_batch = torch.from_numpy(X_np).float().to(device)
                y_mixed = torch.from_numpy(y_mixed).float().to(device)
                use_batch_aug = True
            elif use_mixup and device.type == "cpu":
                X_np = X_batch.cpu().numpy()
                X_np, y_mixed = mixup_data(X_np, y_batch_np, alpha=mixup_alpha, num_classes=num_classes)
                X_batch = torch.from_numpy(X_np).float().to(device)
                y_mixed = torch.from_numpy(y_mixed).float().to(device)
                use_batch_aug = True

            optimizer.zero_grad()

            # AMP 混合精度前向传播
            if amp_manager is not None:
                with amp_manager.autocast_context():
                    outputs = model(X_batch)
                    if use_batch_aug:
                        loss = criterion(outputs, y_mixed)
                    else:
                        loss = criterion(outputs, y_batch)
            else:
                outputs = model(X_batch)
                if use_batch_aug:
                    loss = criterion(outputs, y_mixed)
                else:
                    loss = criterion(outputs, y_batch)

            # AMP-aware backward + optimizer step
            if amp_manager is not None:
                amp_manager.backward(loss)
            else:
                loss.backward()

            # 计算梯度范数
            batch_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    batch_grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm_sum += batch_grad_norm ** 0.5
            num_batches += 1

            if amp_manager is not None:
                amp_manager.optimizer_step(optimizer)
            else:
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            if use_batch_aug:
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        avg_grad_norm = grad_norm_sum / max(num_batches, 1)
        return avg_loss, acc, avg_grad_norm

    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int,
    ) -> Tuple[float, float, float, float, float, List[List[int]], List[int], List[int], List[List[float]], List[float], List[float], List[float]]:
        """验证，返回 (val_loss, val_acc, precision, recall, f1, confusion_matrix, all_preds, all_labels, all_probs, per_class_p, per_class_r, per_class_f1)"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y_batch.cpu().numpy().tolist())
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy().tolist())

        n = len(all_labels)
        avg_loss = total_loss / max(n, 1)
        acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / max(n, 1)

        # 计算宏平均 precision / recall / f1 以及 per-class 指标
        precision, recall, f1, per_class_p, per_class_r, per_class_f1 = self._compute_macro_metrics(all_preds, all_labels, num_classes)

        # 计算混淆矩阵
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        for p, l in zip(all_preds, all_labels):
            if 0 <= l < num_classes and 0 <= p < num_classes:
                confusion_matrix[l][p] += 1

        return avg_loss, acc, precision, recall, f1, confusion_matrix, all_preds, all_labels, all_probs, per_class_p, per_class_r, per_class_f1

    @staticmethod
    def _compute_macro_metrics(
        preds: List[int],
        labels: List[int],
        num_classes: int,
    ) -> Tuple[float, float, float, List[float], List[float], List[float]]:
        """计算宏平均 Precision / Recall / F1，同时返回每类指标"""
        precisions = []
        recalls = []
        f1s = []
        per_class_p = []
        per_class_r = []
        per_class_f1 = []
        for c in range(num_classes):
            tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
            fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
            fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            per_class_p.append(round(p, 4))
            per_class_r.append(round(r, 4))
            per_class_f1.append(round(f, 4))
        macro_p = sum(precisions) / num_classes if num_classes > 0 else 0.0
        macro_r = sum(recalls) / num_classes if num_classes > 0 else 0.0
        macro_f1 = (
            2 * macro_p * macro_r / (macro_p + macro_r)
            if (macro_p + macro_r) > 0
            else 0.0
        )
        return macro_p, macro_r, macro_f1, per_class_p, per_class_r, per_class_f1

    # ============================================================
    # 可视化数据提取（不影响训练核心逻辑）
    # ============================================================

    def _extract_visualization_data(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        all_preds: List[int],
        num_classes: int,
    ) -> Dict[str, Any]:
        """提取可视化数据：激活分布、特征图、预测分布

        在验证后额外执行一次前向传播，不影响训练流程。
        """
        model.eval()
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        # 注册前向钩子，捕获卷积层和全连接层输出
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                def make_hook(layer_name: str):
                    def hook(mod, inp, out):
                        captured[layer_name] = out.detach().cpu()
                    return hook
                h = module.register_forward_hook(make_hook(name))
                handles.append(h)

        try:
            with torch.no_grad():
                # 取第一个样本做前向传播
                model(sample_input[:1].to(next(model.parameters()).device))

            # 分类层
            conv_names = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
            fc_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

            # 构建 layer_activations
            layer_activations: Dict[str, List[float]] = {"input": [], "conv1": [], "conv2": [], "fc": []}

            # input: 使用输入数据采样
            input_flat = sample_input[:1].flatten().cpu()
            layer_activations["input"] = self._sample_and_normalize(input_flat)

            # conv1: 第一个卷积层
            if len(conv_names) >= 1 and conv_names[0] in captured:
                layer_activations["conv1"] = self._sample_and_normalize(captured[conv_names[0]].flatten())

            # conv2: 第二个卷积层
            if len(conv_names) >= 2 and conv_names[1] in captured:
                layer_activations["conv2"] = self._sample_and_normalize(captured[conv_names[1]].flatten())

            # fc: 第一个全连接层
            if len(fc_names) >= 1 and fc_names[0] in captured:
                layer_activations["fc"] = self._sample_and_normalize(captured[fc_names[0]].flatten())

            # 构建特征图（仅从第一个卷积层，取前4个通道，降采样到8x8）
            feature_maps: List[List[List[float]]] = []
            if len(conv_names) >= 1 and conv_names[0] in captured:
                conv_out = captured[conv_names[0]]
                if conv_out.dim() == 4:
                    import torch.nn.functional as F
                    num_channels = min(4, conv_out.shape[1])
                    for c in range(num_channels):
                        fm = conv_out[0, c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        fm_resized = F.adaptive_avg_pool2d(fm, (8, 8))
                        fm_arr = fm_resized.squeeze().numpy()
                        # 归一化到 [0, 1]
                        if fm_arr.max() > fm_arr.min():
                            fm_arr = (fm_arr - fm_arr.min()) / (fm_arr.max() - fm_arr.min())
                        feature_maps.append([[round(float(v), 4) for v in row] for row in fm_arr])

            # 预测分布
            pred_dist = [0] * num_classes
            for p in all_preds:
                if 0 <= p < num_classes:
                    pred_dist[p] += 1
            total = sum(pred_dist)
            if total > 0:
                pred_dist = [round(c / total, 4) for c in pred_dist]

            return {
                "layer_activations": layer_activations,
                "feature_maps": feature_maps,
                "prediction_distribution": pred_dist,
            }
        except Exception as e:
            logger.warning(f"提取可视化数据失败（不影响训练）: {e}")
            return {
                "layer_activations": {"input": [], "conv1": [], "conv2": [], "fc": []},
                "feature_maps": [],
                "prediction_distribution": [0] * num_classes,
            }
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def _sample_and_normalize(tensor: torch.Tensor, n: int = 100) -> List[float]:
        """采样并归一化激活值到 [0, 1]"""
        if tensor.numel() > n:
            indices = torch.randperm(tensor.numel())[:n]
            tensor = tensor[indices]
        values = tensor.numpy().tolist()
        if not values:
            return []
        min_v, max_v = min(values), max(values)
        if max_v > min_v:
            return [round((v - min_v) / (max_v - min_v), 4) for v in values]
        return [0.5] * len(values)

    # ============================================================
    # 数据库操作
    # ============================================================

    def _get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验信息"""
        from app.services.experiment_service import ExperimentService
        return ExperimentService().get_experiment(experiment_id)

    def _update_experiment(self, experiment_id: str, **kwargs):
        """更新实验字段"""
        from app.services.experiment_service import ExperimentService
        ExperimentService().update_experiment(experiment_id, **kwargs)

    def _clear_old_metrics(self, experiment_id: str):
        """根因修复：清理同一实验的旧指标，避免新旧数据拼接导致曲线断崖"""
        db = SessionLocal()
        try:
            deleted = db.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            ).delete()
            db.commit()
            if deleted > 0:
                logger.info(f"[训练 {experiment_id}] 已清理 {deleted} 条旧指标")
        except Exception as e:
            logger.error(f"清理旧指标失败: {e}")
            db.rollback()
        finally:
            db.close()

    def _save_metric(self, experiment_id: str, metric: Dict[str, Any]):
        """保存单条指标到数据库"""
        max_retries = 3
        for attempt in range(max_retries):
            db = SessionLocal()
            try:
                extra = metric.get("extra_data")
                m = ExperimentMetric(
                    experiment_id=experiment_id,
                    step=metric.get("step", 0),
                    epoch=metric.get("epoch", 0),
                    loss=metric.get("loss", 0.0),
                    accuracy=metric.get("accuracy", 0.0),
                    val_loss=metric.get("val_loss", 0.0),
                    val_accuracy=metric.get("val_accuracy", 0.0),
                    learning_rate=metric.get("learning_rate", 0.0),
                    batch_size=metric.get("batch_size", 0),
                    metric_type=metric.get("metric_type", "training"),
                    extra_data=extra,
                )
                db.add(m)
                db.commit()
                return
            except Exception as e:
                logger.warning(f"保存指标失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                try:
                    db.rollback()
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        logger.error(f"保存指标最终失败 (experiment_id={experiment_id}, step={metric.get('step', 0)})")
