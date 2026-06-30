"""
实验服务层
处理实验的增删改查、状态管理、指标存储
"""
import json
import uuid
import csv
import io
import logging
from typing import Optional, Dict, List, Any, Tuple
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import SessionLocal
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric

logger = logging.getLogger(__name__)


class ExperimentService:
    """实验服务"""

    _instance: Optional['ExperimentService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache: Dict[str, Dict] = {}

    def _get_db(self) -> Session:
        return SessionLocal()

    def _to_dict(self, exp: Experiment) -> Dict[str, Any]:
        return {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "description": exp.description or "",
            "model_id": exp.model_id,
            "model_name": exp.model_name,
            "model_architecture": json.loads(exp.model_architecture) if exp.model_architecture else None,
            "status": exp.status,
            "total_params": exp.total_params,
            "layer_count": exp.layer_count,
            "best_accuracy": exp.best_accuracy,
            "final_loss": exp.final_loss,
            "total_epochs": exp.total_epochs,
            "current_step": exp.current_step,
            "hyperparams": json.loads(exp.hyperparams) if exp.hyperparams else {},
            "config": json.loads(exp.config) if exp.config else {},
            "tags": exp.tags.split(",") if exp.tags else [],
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        }

    def create_experiment(
        self,
        name: str,
        description: str = "",
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_architecture: Optional[Dict] = None,
        hyperparams: Optional[Dict] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        total_params: int = 0,
        layer_count: int = 0,
        status: str = "draft",
        best_accuracy: float = 0.0,
        final_loss: float = 0.0,
        total_epochs: int = 0,
        current_step: int = 0,
    ) -> Dict[str, Any]:
        experiment_id = str(uuid.uuid4())

        db = self._get_db()
        try:
            exp = Experiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                model_id=model_id,
                model_name=model_name,
                model_architecture=json.dumps(model_architecture) if model_architecture else None,
                status=status,
                total_params=total_params,
                layer_count=layer_count,
                best_accuracy=best_accuracy,
                final_loss=final_loss,
                total_epochs=total_epochs,
                current_step=current_step,
                hyperparams=json.dumps(hyperparams) if hyperparams else None,
                config=json.dumps(config) if config else None,
                tags=",".join(tags) if tags else "",
            )
            db.add(exp)
            db.commit()
            db.refresh(exp)

            result = self._to_dict(exp)
            self._cache[experiment_id] = result
            return result
        finally:
            db.close()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        if experiment_id in self._cache:
            return self._cache[experiment_id]

        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()
            if exp:
                result = self._to_dict(exp)
                self._cache[experiment_id] = result
                return result
            return None
        finally:
            db.close()

    def list_experiments(
        self,
        page: int = 1,
        page_size: int = 50,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        db = self._get_db()
        try:
            query = db.query(Experiment).filter(Experiment.is_deleted == False)

            if status:
                query = query.filter(Experiment.status == status)

            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    Experiment.name.like(search_pattern)
                    | Experiment.description.like(search_pattern)
                    | Experiment.tags.like(search_pattern)
                )

            total = query.count()

            exps = (
                query.order_by(Experiment.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

            items = [self._to_dict(exp) for exp in exps]

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }
        finally:
            db.close()

    def update_experiment(
        self,
        experiment_id: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()

            if not exp:
                db.close()
                return None

            for key, value in kwargs.items():
                if key == "model_architecture" and value is not None:
                    exp.model_architecture = json.dumps(value)
                elif key == "hyperparams" and value is not None:
                    exp.hyperparams = json.dumps(value)
                elif key == "config" and value is not None:
                    exp.config = json.dumps(value)
                elif key == "tags" and value is not None:
                    exp.tags = ",".join(value) if isinstance(value, list) else str(value)
                elif hasattr(exp, key):
                    setattr(exp, key, value)

            exp.updated_at = datetime.utcnow()

            try:
                db.commit()
                db.refresh(exp)
            except Exception as commit_err:
                logger.warning(f"update_experiment commit失败，尝试回滚后重试: {commit_err}")
                try:
                    db.rollback()
                except Exception:
                    pass
                db.close()
                db = self._get_db()
                exp = db.query(Experiment).filter(
                    Experiment.experiment_id == experiment_id,
                    Experiment.is_deleted == False,
                ).first()
                if not exp:
                    return None
                for key, value in kwargs.items():
                    if key == "model_architecture" and value is not None:
                        exp.model_architecture = json.dumps(value)
                    elif key == "hyperparams" and value is not None:
                        exp.hyperparams = json.dumps(value)
                    elif key == "config" and value is not None:
                        exp.config = json.dumps(value)
                    elif key == "tags" and value is not None:
                        exp.tags = ",".join(value) if isinstance(value, list) else str(value)
                    elif hasattr(exp, key):
                        setattr(exp, key, value)
                exp.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(exp)

            result = self._to_dict(exp)
            self._cache[experiment_id] = result
            return result
        finally:
            try:
                db.close()
            except Exception:
                pass

    def delete_experiment(self, experiment_id: str) -> bool:
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()

            if not exp:
                return False

            exp.is_deleted = True
            exp.updated_at = datetime.utcnow()
            db.commit()

            self._cache.pop(experiment_id, None)
            return True
        finally:
            db.close()

    def rename_experiment(self, experiment_id: str, new_name: str):
        """重命名实验，校验名称唯一性（不区分大小写）

        Returns:
            dict: 更新后的实验数据
            None: 实验不存在
            "duplicate": 名称重复
        """
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()
            if not exp:
                return None

            # 名称唯一性校验（不区分大小写，排除自身）
            duplicate = db.query(Experiment).filter(
                Experiment.is_deleted == False,
                Experiment.experiment_id != experiment_id,
                Experiment.name == new_name,
            ).first()
            if duplicate:
                return "duplicate"

            exp.name = new_name
            exp.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(exp)

            result = self._to_dict(exp)
            self._cache[experiment_id] = result
            return result
        finally:
            db.close()

    def batch_delete_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        delete_all: bool = False,
    ) -> int:
        """批量删除实验（软删除）

        Args:
            experiment_ids: 要删除的实验ID列表
            delete_all: 是否删除全部未删除实验（忽略 experiment_ids）

        Returns:
            实际删除的实验数量
        """
        db = self._get_db()
        try:
            query = db.query(Experiment).filter(Experiment.is_deleted == False)

            if not delete_all:
                if not experiment_ids:
                    return 0
                query = query.filter(Experiment.experiment_id.in_(experiment_ids))

            count = query.update(
                {"is_deleted": True, "updated_at": datetime.utcnow()},
                synchronize_session=False,
            )
            db.commit()

            # 清理缓存
            if delete_all:
                self._cache.clear()
            elif experiment_ids:
                for eid in experiment_ids:
                    self._cache.pop(eid, None)

            logger.info(f"批量删除实验: {count} 个")
            return count
        except Exception as e:
            logger.error(f"批量删除实验失败: {e}")
            db.rollback()
            raise
        finally:
            db.close()

    def add_metrics(
        self,
        experiment_id: str,
        metrics: List[Dict[str, Any]],
    ) -> int:
        db = self._get_db()
        try:
            count = 0
            for m in metrics:
                metric = ExperimentMetric(
                    experiment_id=experiment_id,
                    step=m.get("step", 0),
                    epoch=m.get("epoch", 0),
                    loss=m.get("loss", 0.0),
                    accuracy=m.get("accuracy", 0.0),
                    val_loss=m.get("val_loss", 0.0),
                    val_accuracy=m.get("val_accuracy", 0.0),
                    learning_rate=m.get("learning_rate", 0.0),
                    batch_size=m.get("batch_size", 0),
                    metric_type=m.get("metric_type", "training"),
                    extra_data=json.dumps(m.get("extra_data")) if m.get("extra_data") else None,
                )
                db.add(metric)
                count += 1

            db.commit()
            return count
        finally:
            db.close()

    def get_metrics(
        self,
        experiment_id: str,
        metric_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        db = self._get_db()
        try:
            query = db.query(ExperimentMetric).filter(
                ExperimentMetric.experiment_id == experiment_id
            )

            if metric_type:
                query = query.filter(ExperimentMetric.metric_type == metric_type)

            metrics = query.order_by(ExperimentMetric.step.asc()).limit(limit).all()

            result = []
            for m in metrics:
                result.append({
                    "step": m.step,
                    "epoch": m.epoch,
                    "loss": m.loss,
                    "accuracy": m.accuracy,
                    "val_loss": m.val_loss,
                    "val_accuracy": m.val_accuracy,
                    "learning_rate": m.learning_rate,
                    "batch_size": m.batch_size,
                    "metric_type": m.metric_type,
                    "extra_data": json.loads(m.extra_data) if m.extra_data else None,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                })

            return result
        finally:
            db.close()

    def get_summary(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        db = self._get_db()
        try:
            exps = db.query(Experiment).filter(
                Experiment.experiment_id.in_(experiment_ids),
                Experiment.is_deleted == False,
            ).all()

            result = []
            for exp in exps:
                exp_dict = self._to_dict(exp)

                metrics = self.get_metrics(exp.experiment_id, limit=1)
                latest_metric = metrics[-1] if metrics else None

                exp_dict["metrics"] = {
                    "best_accuracy": exp.best_accuracy,
                    "final_loss": exp.final_loss,
                    "total_epochs": exp.total_epochs,
                    "current_step": exp.current_step,
                    "latest_step": latest_metric.get("step", 0) if latest_metric else 0,
                }

                result.append(exp_dict)

            return result
        finally:
            db.close()

    # ============================================================
    # 单实验详情接口 - 返回全量实验数据
    # ============================================================

    def get_experiment_detail(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验完整详情，包含层详情、时序数据、训练日志

        返回结构：
        - basic_info: 基础信息
        - model_config: 模型配置
        - layers: 层详情列表
        - hyperparams: 超参数
        - metrics_summary: 训练指标汇总
        - training_history: 全量训练时序数据
        - training_logs: 训练日志
        """
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()
            if not exp:
                return None

            base = self._to_dict(exp)

            # 1. 基础信息
            basic_info = {
                "experiment_id": base["experiment_id"],
                "name": base["name"],
                "description": base["description"],
                "status": base["status"],
                "tags": base["tags"],
                "created_at": base["created_at"],
                "updated_at": base["updated_at"],
                "remark": base["config"].get("remark", "") if isinstance(base["config"], dict) else "",
            }

            # 2. 模型配置（model_name 兜底，禁止返回空值导致前端显示纯ID）
            architecture = base["model_architecture"] or {}
            _arch_input_shape = architecture.get("input_shape", []) if isinstance(architecture, dict) else []
            _arch_output_shape = architecture.get("output_shape", []) if isinstance(architecture, dict) else []
            if not isinstance(_arch_input_shape, (list, tuple)):
                _arch_input_shape = [_arch_input_shape] if _arch_input_shape is not None else []
            if not isinstance(_arch_output_shape, (list, tuple)):
                _arch_output_shape = [_arch_output_shape] if _arch_output_shape is not None else []
            model_config = {
                "model_type": architecture.get("type", "") if isinstance(architecture, dict) else "",
                "model_name": base["model_name"] or (architecture.get("name", "") if isinstance(architecture, dict) else "") or "未命名模型",
                "total_params": base["total_params"],
                "total_layers": base["layer_count"],
                "input_shape": list(_arch_input_shape),
                "output_shape": list(_arch_output_shape),
            }

            # 3. 层详情列表
            layers: List[Dict[str, Any]] = []
            if isinstance(architecture, dict) and isinstance(architecture.get("layers"), list):
                for layer in architecture["layers"]:
                    if not isinstance(layer, dict):
                        continue
                    # 从 PyTorch 字段推导 input/output shape
                    in_shape = layer.get("inputShape", layer.get("input_shape"))
                    out_shape = layer.get("outputShape", layer.get("output_shape"))
                    if in_shape is None:
                        if layer.get("in_features") is not None:
                            in_shape = [layer["in_features"]]
                        elif layer.get("in_channels") is not None:
                            in_shape = [layer["in_channels"]]
                        else:
                            in_shape = []
                    if out_shape is None:
                        if layer.get("out_features") is not None:
                            out_shape = [layer["out_features"]]
                        elif layer.get("out_channels") is not None:
                            out_shape = [layer["out_channels"]]
                        else:
                            out_shape = []
                    layers.append({
                        "name": layer.get("name", ""),
                        "type": layer.get("type", ""),
                        "params": layer.get("params", 0),
                        "input_shape": in_shape,
                        "output_shape": out_shape,
                        "node_count": layer.get("nodeCount", layer.get("node_count", 0)),
                        "activation": layer.get("activation", ""),
                        "kernel_size": layer.get("kernelSize", layer.get("kernel_size")),
                        "in_features": layer.get("in_features"),
                        "out_features": layer.get("out_features"),
                        "in_channels": layer.get("in_channels"),
                        "out_channels": layer.get("out_channels"),
                        "stride": layer.get("stride"),
                        "padding": layer.get("padding"),
                        "dropout": layer.get("dropout"),
                    })

            # 4. 超参数（合并 hyperparams 与 config 中的训练相关字段）
            hyperparams = base["hyperparams"] or {}
            config = base["config"] or {}
            hyperparams_full = {
                "learning_rate": hyperparams.get("learning_rate", config.get("learning_rate", 0.0)),
                "batch_size": hyperparams.get("batch_size", config.get("batch_size", 0)),
                "optimizer": hyperparams.get("optimizer", config.get("optimizer", "")),
                "total_epochs": base["total_epochs"],
                "random_seed": hyperparams.get("random_seed", hyperparams.get("seed", config.get("random_seed", config.get("seed", 0)))),
                "val_split": hyperparams.get("val_split", config.get("val_split", 0.2)),
                "loss_function": hyperparams.get("loss_function", config.get("loss_function", "")),
                "dataset_name": config.get("dataset_name", hyperparams.get("dataset_name", "")),
                "dataset_version": config.get("dataset_version", hyperparams.get("dataset_version", "")),
            }

            # 5. 训练指标汇总
            # 无值时返回 None，前端做兜底显示（避免 0 被误判为有效值）
            metrics_summary = {
                "best_accuracy": base["best_accuracy"],
                "final_loss": base["final_loss"],
                "best_epoch": config.get("best_epoch") if config.get("best_epoch") else None,
                "training_duration": config.get("training_duration") if config.get("training_duration") else None,
            }

            # 6. 全量训练时序数据
            training_history = self._build_training_history(db, experiment_id)

            # 7. 训练日志
            training_logs = config.get("training_logs", [])
            if not isinstance(training_logs, list):
                training_logs = []

            return {
                **base,  # 保持向后兼容：原有字段全部保留
                "basic_info": basic_info,
                "model_config": model_config,
                "layers": layers,
                "hyperparams": hyperparams_full,
                "metrics_summary": metrics_summary,
                "training_history": training_history,
                "training_logs": training_logs,
            }
        finally:
            db.close()

    def _build_training_history(
        self,
        db: Session,
        experiment_id: str,
    ) -> List[Dict[str, Any]]:
        """构建全量训练时序数据，合并 metrics 表与 extra_data 中的扩展字段

        每条记录包含：epoch, train_loss, val_loss, train_acc, val_acc,
                      precision, recall, f1, gradient_norm, weight_norm,
                      per_class_precision, per_class_recall, per_class_f1, confusion_matrix
        """
        metrics = (
            db.query(ExperimentMetric)
            .filter(ExperimentMetric.experiment_id == experiment_id)
            .order_by(ExperimentMetric.step.asc())
            .all()
        )

        history: List[Dict[str, Any]] = []
        for m in metrics:
            extra = {}
            if m.extra_data:
                try:
                    extra = json.loads(m.extra_data) or {}
                except (json.JSONDecodeError, TypeError):
                    extra = {}

            history.append({
                "epoch": m.epoch,
                "step": m.step,
                "train_loss": round(float(m.loss), 4) if m.loss is not None else 0.0,
                "val_loss": round(float(m.val_loss), 4) if m.val_loss is not None else 0.0,
                "train_acc": round(float(m.accuracy), 4) if m.accuracy is not None else 0.0,
                "val_acc": round(float(m.val_accuracy), 4) if m.val_accuracy is not None else 0.0,
                "precision": round(float(extra.get("precision", 0.0)), 4),
                "recall": round(float(extra.get("recall", 0.0)), 4),
                "f1": round(float(extra.get("f1", extra.get("f1_score", 0.0))), 4),
                "learning_rate": float(m.learning_rate) if m.learning_rate else 0.0,
                "gradient_norm": round(float(extra.get("gradient_norm", extra.get("grad_norm", 0.0))), 4),
                "weight_norm": round(float(extra.get("weight_norm", 0.0)), 4),
                "per_class_precision": extra.get("per_class_precision", []),
                "per_class_recall": extra.get("per_class_recall", []),
                "per_class_f1": extra.get("per_class_f1", []),
                "confusion_matrix": extra.get("confusion_matrix", []),
                "prediction_distribution": extra.get("prediction_distribution", []),
            })
        return history

    # ============================================================
    # CSV 批量导出
    # ============================================================

    # CSV 导出字段定义：(CSV列名, 取值函数)
    _CSV_FIELDS: List[Tuple[str, Any]] = [
        ("实验名称", lambda exp, cfg, hp: exp.name or ""),
        ("状态", lambda exp, cfg, hp: exp.status or ""),
        ("模型类型", lambda exp, cfg, hp: _extract_model_type(exp)),
        ("模型名称", lambda exp, cfg, hp: exp.model_name or ""),
        ("总参数量", lambda exp, cfg, hp: exp.total_params or 0),
        ("总层数", lambda exp, cfg, hp: exp.layer_count or 0),
        ("最佳准确率", lambda exp, cfg, hp: round(float(exp.best_accuracy or 0.0), 4)),
        ("最终Train Loss", lambda exp, cfg, hp: _get_final_train_loss(exp, cfg)),
        ("最终Val Loss", lambda exp, cfg, hp: _get_final_val_loss(exp, cfg)),
        ("学习率", lambda exp, cfg, hp: round(float(hp.get("learning_rate", 0.0)), 4)),
        ("批次大小", lambda exp, cfg, hp: hp.get("batch_size", 0)),
        ("优化器", lambda exp, cfg, hp: hp.get("optimizer", "")),
        ("训练轮次", lambda exp, cfg, hp: exp.total_epochs or 0),
        ("随机种子", lambda exp, cfg, hp: hp.get("seed", hp.get("random_seed", 0))),
        ("数据集名称", lambda exp, cfg, hp: cfg.get("dataset_name", "")),
        ("Precision", lambda exp, cfg, hp: round(float(cfg.get("precision", 0.0)), 4)),
        ("Recall", lambda exp, cfg, hp: round(float(cfg.get("recall", 0.0)), 4)),
        ("F1", lambda exp, cfg, hp: round(float(cfg.get("f1", cfg.get("f1_score", 0.0))), 4)),
        ("创建时间", lambda exp, cfg, hp: _format_datetime(exp.created_at)),
        ("更新时间", lambda exp, cfg, hp: _format_datetime(exp.updated_at)),
        ("标签", lambda exp, cfg, hp: _format_tags(exp.tags)),
        ("备注", lambda exp, cfg, hp: cfg.get("remark", "")),
    ]

    def export_experiments_csv(
        self,
        experiment_ids: Optional[List[str]] = None,
    ) -> str:
        """导出实验数据为 CSV 字符串

        Args:
            experiment_ids: 要导出的实验ID列表；为空时导出全部未删除实验

        Returns:
            UTF-8 BOM 编码的 CSV 字符串（Excel 兼容）
        """
        db = self._get_db()
        try:
            query = db.query(Experiment).filter(Experiment.is_deleted == False)
            if experiment_ids:
                query = query.filter(Experiment.experiment_id.in_(experiment_ids))
            exps = query.order_by(Experiment.created_at.desc()).all()

            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            # 写入表头
            headers = [field[0] for field in self._CSV_FIELDS]
            writer.writerow(headers)

            # 写入数据行
            for exp in exps:
                cfg = json.loads(exp.config) if exp.config else {}
                hp = json.loads(exp.hyperparams) if exp.hyperparams else {}
                row = [field[1](exp, cfg, hp) for field in self._CSV_FIELDS]
                writer.writerow(row)

            csv_content = output.getvalue()
            output.close()
            # 加 UTF-8 BOM 头，确保 Excel 正确识别编码
            return "\ufeff" + csv_content
        finally:
            db.close()


    def export_experiment_metrics_csv(self, experiment_id: str) -> str:
        """导出单个实验的逐epoch训练指标为CSV（科研用，包含所有时序指标）

        包含列：epoch, train_loss, val_loss, train_acc, val_acc, precision, recall, f1,
               learning_rate, gradient_norm, weight_norm
        """
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == experiment_id,
                Experiment.is_deleted == False,
            ).first()
            if not exp:
                raise ValueError(f"实验 {experiment_id} 不存在")

            history = self._build_training_history(db, experiment_id)

            output = io.StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

            writer.writerow([
                "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                "precision", "recall", "f1", "learning_rate", "gradient_norm", "weight_norm"
            ])

            for h in history:
                writer.writerow([
                    h["epoch"], h["train_loss"], h["val_loss"], h["train_acc"], h["val_acc"],
                    h["precision"], h["recall"], h["f1"],
                    round(h["learning_rate"], 6), h["gradient_norm"], h["weight_norm"],
                ])

            return "\ufeff" + output.getvalue()
        finally:
            db.close()

    def export_experiment_json(self, experiment_id: str) -> Dict[str, Any]:
        """导出单个实验的完整数据为JSON（包含配置、模型结构、全量时序指标、日志、混淆矩阵）"""
        detail = self.get_experiment_detail(experiment_id)
        if not detail:
            raise ValueError(f"实验 {experiment_id} 不存在")
        return detail

    # ============================================================
    # 自定义实验模板管理
    # ============================================================

    def list_templates(self, template_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取自定义实验模板列表"""
        db = self._get_db()
        try:
            query = db.query(Experiment).filter(
                Experiment.is_deleted == False,
                Experiment.status == "template",
            )
            if template_type:
                # 模板类型存储在 config.template_type 中
                exps = query.all()
                result = []
                for exp in exps:
                    cfg = json.loads(exp.config) if exp.config else {}
                    if cfg.get("template_type") == template_type:
                        result.append(self._template_to_dict(exp, cfg))
                return result
            else:
                exps = query.order_by(Experiment.updated_at.desc()).all()
                return [self._template_to_dict(exp, json.loads(exp.config) if exp.config else {}) for exp in exps]
        finally:
            db.close()

    def save_template(
        self,
        name: str,
        description: str,
        template_type: str,
        configs: List[Dict[str, Any]],
        comparison_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """保存自定义实验模板"""
        template_id = str(uuid.uuid4())
        db = self._get_db()
        try:
            template_config = {
                "template_type": template_type,
                "configs": configs,
                "comparison_metrics": comparison_metrics or ["val_acc", "val_loss", "best_accuracy"],
            }
            exp = Experiment(
                experiment_id=template_id,
                name=name,
                description=description,
                status="template",
                config=json.dumps(template_config),
            )
            db.add(exp)
            db.commit()
            db.refresh(exp)
            return self._template_to_dict(exp, template_config)
        finally:
            db.close()

    def delete_template(self, template_id: str) -> bool:
        """删除自定义实验模板（软删除）"""
        db = self._get_db()
        try:
            exp = db.query(Experiment).filter(
                Experiment.experiment_id == template_id,
                Experiment.status == "template",
                Experiment.is_deleted == False,
            ).first()
            if not exp:
                return False
            exp.is_deleted = True
            exp.updated_at = datetime.utcnow()
            db.commit()
            return True
        finally:
            db.close()

    def _template_to_dict(self, exp: Experiment, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "template_id": exp.experiment_id,
            "name": exp.name,
            "description": exp.description or "",
            "template_type": cfg.get("template_type", "comparison"),
            "configs": cfg.get("configs", []),
            "comparison_metrics": cfg.get("comparison_metrics", ["val_acc", "val_loss", "best_accuracy"]),
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
        }


# ============================================================
# CSV 导出辅助函数（模块级，避免 self 闭包开销）
# ============================================================

def _extract_model_type(exp: Experiment) -> str:
    """从 model_architecture JSON 中提取模型类型"""
    if not exp.model_architecture:
        return ""
    try:
        arch = json.loads(exp.model_architecture)
        if isinstance(arch, dict):
            return arch.get("type", "")
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _get_final_train_loss(exp: Experiment, cfg: Dict[str, Any]) -> float:
    """获取最终训练 Loss"""
    val = cfg.get("final_train_loss")
    if val is not None:
        return round(float(val), 4)
    return round(float(exp.final_loss or 0.0), 4)


def _get_final_val_loss(exp: Experiment, cfg: Dict[str, Any]) -> float:
    """获取最终验证 Loss"""
    val = cfg.get("final_val_loss")
    if val is not None:
        return round(float(val), 4)
    # 回退：使用 final_loss 作为近似值
    return round(float(exp.final_loss or 0.0), 4)


def _format_datetime(dt: Optional[datetime]) -> str:
    """格式化日期为 YYYY-MM-DD HH:mm:ss"""
    if not dt:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _format_tags(tags_str: Optional[str]) -> str:
    """格式化标签：数据库中以英文逗号存储，输出保持英文逗号分隔"""
    if not tags_str:
        return ""
    # 去除空白项，统一用英文逗号分隔
    items = [t.strip() for t in tags_str.split(",") if t.strip()]
    return ",".join(items)
