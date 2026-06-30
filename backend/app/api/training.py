"""
训练引擎 API 路由
前缀 /api/training
"""
import os
import json
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.core.security import sanitize_path_id

from app.services.training_service import TrainingService
from app.core.config import settings

router = APIRouter()
training_service = TrainingService()


class StartTrainingRequest(BaseModel):
    """启动训练请求"""
    dataset_id: Optional[str] = Field(None, description="数据集 ID，覆盖实验配置")
    hyperparams: Optional[Dict[str, Any]] = Field(
        None,
        description="超参数覆盖：learning_rate / batch_size / optimizer / epochs / random_seed / val_split / loss_function",
    )
    model_structure_config: Optional[Dict[str, Any]] = Field(
        None,
        description="模型结构配置：channel_list / attention / use_bn / use_dropout 等",
        alias="model_config",
    )

    class Config:
        populate_by_name = True


@router.post("/start/{experiment_id}")
async def start_training(experiment_id: str, request: StartTrainingRequest):
    """启动训练任务

    - 校验实验、数据集是否就绪
    - 异步后台执行训练，不阻塞响应
    - 成功返回任务已提交，状态更新为 running
    """
    try:
        result = training_service.start_training(
            experiment_id=experiment_id,
            dataset_id=request.dataset_id,
            hyperparams_override=request.hyperparams,
            model_config_override=request.model_structure_config,
        )
        return {
            "code": 200,
            "message": "训练任务已启动",
            "data": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


@router.post("/stop/{experiment_id}")
async def stop_training(experiment_id: str):
    """停止正在运行的训练任务（优雅退出）

    - 设置停止标志位，训练循环在当前 epoch 结束后退出
    - 已保存的模型与指标数据保留
    """
    try:
        result = training_service.stop_training(experiment_id)
        return {
            "code": 200,
            "message": "停止信号已发送",
            "data": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止训练失败: {str(e)}")


@router.get("/status/{experiment_id}")
async def get_training_status(experiment_id: str):
    """获取训练实时状态

    返回：status、current_epoch、total_epochs、latest_metrics、elapsed_seconds
    """
    result = training_service.get_status(experiment_id)
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.get("/logs/{experiment_id}")
async def get_training_logs(
    experiment_id: str,
    since: int = Query(0, ge=0, description="从第几条日志开始返回（增量获取）"),
):
    """获取训练日志

    - 支持 since 参数增量获取，避免重复传输
    """
    result = training_service.get_logs(experiment_id, since=since)
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.get("/metrics/{experiment_id}")
async def get_training_metrics(experiment_id: str):
    """获取全量训练指标时序数据

    - 返回所有已完成 epoch 的完整指标序列
    - 与原有实验指标接口格式兼容
    """
    result = training_service.get_metrics(experiment_id)
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


# ============================================================
# CNN 可视化数据 API（特征图 / 卷积核 / 注意力权重）
# ============================================================

@router.get("/visualizations/{experiment_id}")
async def get_cnn_visualizations(experiment_id: str):
    """获取训练完成后保存的CNN可视化数据

    返回特征图、卷积核权重、注意力热力图数据（前端期望格式）
    """
    experiment_id = sanitize_path_id(experiment_id)
    vis_dir = settings.MODEL_DIR / experiment_id / "visualizations"
    if not vis_dir.exists():
        return {
            "code": 200,
            "message": "暂无可视化数据（可能训练尚未完成或非图像模型）",
            "data": {"available": False},
        }

    result = {"available": True, "feature_maps": {}, "conv_kernels": {}, "attention": None}

    fm_path = vis_dir / "feature_maps.json"
    if fm_path.exists():
        with open(fm_path, "r", encoding="utf-8") as f:
            raw_fm = json.load(f)
        for layer_name, layer_data in raw_fm.items():
            if isinstance(layer_data, dict) and "channels" in layer_data:
                result["feature_maps"][layer_name] = layer_data["channels"]
            elif isinstance(layer_data, list):
                result["feature_maps"][layer_name] = layer_data

    kw_path = vis_dir / "kernels.json"
    if kw_path.exists():
        with open(kw_path, "r", encoding="utf-8") as f:
            raw_kw = json.load(f)
        for layer_name, layer_data in raw_kw.items():
            if isinstance(layer_data, dict) and "kernels" in layer_data:
                flat_kernels = []
                for oc_kernels in layer_data["kernels"]:
                    if isinstance(oc_kernels, list) and len(oc_kernels) > 0:
                        flat_kernels.append(oc_kernels[0])
                result["conv_kernels"][layer_name] = flat_kernels
            elif isinstance(layer_data, list):
                result["conv_kernels"][layer_name] = layer_data

    attn_path = vis_dir / "attention_weights.json"
    if attn_path.exists():
        with open(attn_path, "r", encoding="utf-8") as f:
            raw_attn = json.load(f)
        for layer_name, attn_data in raw_attn.items():
            if isinstance(attn_data, dict):
                for key, val in attn_data.items():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                        result["attention"] = val
                        result["attention_weights"] = val
                        break
            elif isinstance(attn_data, list) and len(attn_data) > 0 and isinstance(attn_data[0], (int, float)):
                result["attention"] = attn_data
                result["attention_weights"] = attn_data

    return {"code": 200, "message": "获取成功", "data": result}


# ============================================================
# 消融实验 / 对比实验 API
# ============================================================

class AblationExperimentRequest(BaseModel):
    """消融实验/对比实验请求：批量运行多个实验配置并对比结果"""
    base_experiment_id: Optional[str] = Field(None, description="基础实验ID（复用数据集和基础配置）")
    dataset_id: str = Field(..., description="数据集ID")
    name_prefix: str = Field("ablation", description="实验名称前缀")
    epochs: int = Field(10, ge=1, le=200, description="每个配置训练轮次")
    batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(0.001, ge=1e-5, le=1.0)
    val_split: float = Field(0.2, gt=0, lt=1)
    random_seed: int = Field(42)

    # 对比配置列表
    configs: List[Dict[str, Any]] = Field(
        ...,
        description="待对比的模型配置列表，每项包含: name, channels, attention, use_bn, use_dropout, use_residual"
    )


@router.post("/ablation/run")
async def run_ablation_experiment(request: AblationExperimentRequest):
    """批量运行消融/对比实验

    传入多个模型配置（不同注意力机制/通道数/组件开关），后端依次训练并返回各实验ID。
    前端可轮询各实验训练状态，完成后汇总对比结果。
    """
    from app.core.database import SessionLocal
    from app.models.experiment import Experiment
    from app.services.experiment_service import ExperimentService
    import copy

    exp_service = ExperimentService()
    created_ids = []

    for cfg in request.configs:
        cfg_name = cfg.get("name", f"config_{len(created_ids)}")
        model_config = {
            "channels": cfg.get("channels", [32, 64]),
            "attention": cfg.get("attention", "none"),
            "use_bn": cfg.get("use_bn", True),
            "use_dropout": cfg.get("use_dropout", True),
            "dropout_rate": cfg.get("dropout_rate", 0.3),
            "use_residual": cfg.get("use_residual", False),
            "fc_hidden": cfg.get("fc_hidden", 128),
        }
        # 对比实验中自定义的学习率覆盖
        lr = cfg.get("learning_rate", request.learning_rate)
        epochs = cfg.get("epochs", request.epochs)

        exp_name = f"{request.name_prefix}_{cfg_name}"
        exp_config = {
            "dataset_id": request.dataset_id,
            "model_config": model_config,
            "ablation_group": request.name_prefix,
            "ablation_config_name": cfg_name,
        }
        hyperparams = {
            "learning_rate": lr,
            "batch_size": request.batch_size,
            "epochs": epochs,
            "optimizer": cfg.get("optimizer", "adam"),
            "random_seed": request.random_seed,
            "val_split": request.val_split,
            "loss_function": "cross_entropy",
        }

        try:
            exp_result = exp_service.create_experiment(
                name=exp_name,
                description=f"消融实验: {cfg_name}, channels={model_config['channels']}, attention={model_config['attention']}",
                config=exp_config,
                hyperparams=hyperparams,
                status="draft",
            )
            exp_id = exp_result["experiment_id"] if isinstance(exp_result, dict) else exp_result
            created_ids.append({
                "config_name": cfg_name,
                "experiment_id": exp_id,
                "model_config": model_config,
                "learning_rate": lr,
                "epochs": epochs,
            })
            # 启动训练
            training_service.start_training(experiment_id=exp_id, dataset_id=request.dataset_id)
        except Exception as e:
            created_ids.append({
                "config_name": cfg_name,
                "error": str(e),
            })

    return {
        "code": 200,
        "message": f"已提交 {len(created_ids)} 个对比实验",
        "data": {
            "group_name": request.name_prefix,
            "ablation_group": request.name_prefix,
            "experiments": created_ids,
        },
    }


@router.get("/ablation/results/{group_name}")
async def get_ablation_results(group_name: str):
    """获取同一消融实验组所有实验的结果对比"""
    from app.core.database import SessionLocal
    from app.models.experiment import Experiment
    import json as _json

    db = SessionLocal()
    try:
        exps = db.query(Experiment).filter(Experiment.is_deleted == False).all()
        results = []
        for exp in exps:
            cfg = _json.loads(exp.config) if exp.config else {}
            if cfg.get("ablation_group") == group_name:
                hp = _json.loads(exp.hyperparams) if exp.hyperparams else {}
                mc = cfg.get("model_config", {})
                results.append({
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "config_name": cfg.get("ablation_config_name", ""),
                    "status": exp.status,
                    "channels": mc.get("channels", [32, 64]),
                    "attention": mc.get("attention", "none"),
                    "use_bn": mc.get("use_bn", True),
                    "use_dropout": mc.get("use_dropout", True),
                    "use_residual": mc.get("use_residual", False),
                    "learning_rate": hp.get("learning_rate", 0.001),
                    "best_accuracy": exp.best_accuracy,
                    "final_loss": exp.final_loss,
                    "total_params": exp.total_params,
                    "total_epochs": exp.total_epochs,
                })
        results.sort(key=lambda x: (x.get("best_accuracy") or 0), reverse=True)
        return {"code": 200, "message": "获取成功", "data": {"group": group_name, "results": results}}
    finally:
        db.close()


# ============================================================
# 模型导入 API
# ============================================================

class ModelImportRequest(BaseModel):
    """模型导入请求"""
    model_name: str = Field(..., description="模型名称")
    model_format: str = Field("pytorch", description="模型格式: pytorch / onnx")
    dataset_type: str = Field("image_folder", description="数据集类型")
    feature_shape: str = Field("1x28x28", description="特征维度")
    num_classes: int = Field(10, description="类别数")
    model_structure_config: Optional[Dict[str, Any]] = Field(None, description="模型结构配置")
    description: Optional[str] = Field(None, description="模型描述")


@router.post("/import-model")
async def import_model(
    model_name: str = Query(..., description="模型名称"),
    model_format: str = Query("pytorch", description="模型格式: pytorch / onnx"),
    dataset_type: str = Query("image_folder", description="数据集类型"),
    feature_shape: str = Query("1x28x28", description="特征维度"),
    num_classes: int = Query(10, description="类别数"),
):
    """导入外部模型文件

    支持从 uploads/models/ 目录导入已训练的模型文件（.pt/.pth/.onnx）
    返回模型结构摘要和参数量信息
    """
    from app.ml.model_builder import build_model, count_parameters
    from app.ml.model_builder import MODEL_ARCHITECTURES

    sanitized_name = sanitize_path_id(model_name)
    model_dir = settings.MODEL_DIR / sanitized_name
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"模型目录不存在: {sanitized_name}")

    # 查找模型文件
    model_files = []
    for ext in (".pt", ".pth", ".onnx"):
        model_files.extend(model_dir.glob(f"*.{ext.lstrip('.')}"))
    if not model_files:
        raise HTTPException(status_code=404, detail="模型目录中未找到模型文件 (.pt/.pth/.onnx)")

    model_path = model_files[0]
    model_format = model_format.lower()

    try:
        # 构建模型结构
        model = build_model(dataset_type, feature_shape, num_classes)
        total_params = count_parameters(model)

        # 加载权重（仅 PyTorch 格式）
        if model_format in ("pytorch", "pt", "pth"):
            import torch
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            loaded = True
        elif model_format == "onnx":
            loaded = False  # ONNX 不需要加载权重到 PyTorch 模型
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型格式: {model_format}")

        # 提取模型结构信息
        layer_info = []
        for name, module in model.named_modules():
            if name == "" or len(list(module.children())) > 0:
                continue
            info = {"name": name, "type": module.__class__.__name__}
            if hasattr(module, "in_channels"):
                info["in_channels"] = module.in_channels
            if hasattr(module, "out_channels"):
                info["out_channels"] = module.out_channels
            if hasattr(module, "in_features"):
                info["in_features"] = module.in_features
            if hasattr(module, "out_features"):
                info["out_features"] = module.out_features
            params = sum(p.numel() for p in module.parameters(recurse=False))
            info["params"] = params
            layer_info.append(info)

        return {
            "code": 200,
            "message": "模型导入成功",
            "data": {
                "model_name": model_name,
                "model_path": str(model_path),
                "model_format": model_format,
                "total_params": total_params,
                "layer_count": len(layer_info),
                "layers": layer_info,
                "model_class": model.__class__.__name__,
                "weight_loaded": loaded,
                "available_architectures": list(MODEL_ARCHITECTURES.keys()),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型导入失败: {str(e)}")


# ============================================================
# 模型 / 损失函数 元数据 API
# ============================================================

@router.get("/meta/attention-types")
async def get_attention_types():
    """获取所有支持的注意力机制类型"""
    from app.ml.attention import ATTENTION_REGISTRY
    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "attention_types": [
                {
                    "value": k,
                    "label": v.__name__ if v else "无注意力",
                    "description": v.__doc__.split("\n")[0].strip() if v and v.__doc__ else "不使用注意力机制",
                }
                for k, v in ATTENTION_REGISTRY.items()
            ],
        },
    }


@router.get("/meta/loss-functions")
async def get_loss_functions():
    """获取所有支持的损失函数"""
    from app.ml.losses import list_loss_functions, LOSS_REGISTRY
    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "loss_functions": [
                {
                    "value": k,
                    "label": v.__name__,
                    "description": v.__doc__.split("\n")[0].strip() if v.__doc__ else "",
                }
                for k, v in LOSS_REGISTRY.items()
            ],
        },
    }


@router.get("/meta/model-architectures")
async def get_model_architectures():
    """获取所有支持的模型架构"""
    from app.ml.model_builder import MODEL_ARCHITECTURES
    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "architectures": list(MODEL_ARCHITECTURES.keys()),
        },
    }


@router.get("/meta/activations")
async def get_activations():
    """获取所有支持的激活函数"""
    from app.ml.model_builder import ConfigurableCNN
    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "activations": [
                {"value": k, "label": v.__name__}
                for k, v in ConfigurableCNN.ACTIVATIONS.items()
            ],
        },
    }
