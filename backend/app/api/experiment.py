"""
实验管理 API 路由
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.services.experiment_service import ExperimentService

router = APIRouter()
experiment_service = ExperimentService()


class CreateExperimentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = ""
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    model_architecture: Optional[Dict[str, Any]] = None
    hyperparams: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    total_params: int = 0
    layer_count: int = 0
    status: str = "draft"
    best_accuracy: float = 0.0
    final_loss: float = 0.0
    total_epochs: int = 0
    current_step: int = 0


class UpdateExperimentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    hyperparams: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    best_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    total_epochs: Optional[int] = None
    current_step: Optional[int] = None


class AddMetricsRequest(BaseModel):
    metrics: List[Dict[str, Any]]


class BatchExportRequest(BaseModel):
    """批量导出 CSV 请求体"""
    experiment_ids: Optional[List[str]] = Field(
        None,
        description="要导出的实验ID列表；为空时导出全部未删除实验",
    )


class RenameExperimentRequest(BaseModel):
    """重命名实验请求体"""
    name: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="新实验名称，2-50字符，支持中文、英文、数字、下划线、连字符、空格",
    )


class BatchDeleteRequest(BaseModel):
    """批量删除实验请求体"""
    experiment_ids: Optional[List[str]] = Field(
        None,
        description="要删除的实验ID列表；delete_all=true时忽略",
    )
    delete_all: bool = Field(
        False,
        description="是否删除全部未删除实验",
    )


@router.get("")
async def list_experiments(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = None,
    search: Optional[str] = None,
):
    """获取实验列表"""
    result = experiment_service.list_experiments(
        page=page,
        page_size=page_size,
        status=status,
        search=search,
    )
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.post("")
async def create_experiment(request: CreateExperimentRequest):
    """创建新实验"""
    try:
        result = experiment_service.create_experiment(
            name=request.name,
            description=request.description,
            model_id=request.model_id,
            model_name=request.model_name,
            model_architecture=request.model_architecture,
            hyperparams=request.hyperparams,
            config=request.config,
            tags=request.tags,
            total_params=request.total_params,
            layer_count=request.layer_count,
            status=request.status,
            best_accuracy=request.best_accuracy,
            final_loss=request.final_loss,
            total_epochs=request.total_epochs,
            current_step=request.current_step,
        )
        return {
            "code": 200,
            "message": "实验创建成功",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建实验失败: {str(e)}")


@router.get("/summary")
async def get_experiments_summary(
    ids: str = Query(..., description="实验ID列表，逗号分隔"),
):
    """批量获取实验摘要（用于对比）"""
    experiment_ids = [eid.strip() for eid in ids.split(",") if eid.strip()]
    if not experiment_ids:
        raise HTTPException(status_code=400, detail="请提供至少一个实验ID")

    result = experiment_service.get_summary(experiment_ids)
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.post("/batch/export")
async def batch_export_experiments(request: BatchExportRequest):
    """批量导出实验数据为 CSV

    - 支持单实验导出与批量实验导出，字段保持一致
    - 数值统一保留4位小数
    - 日期统一输出 YYYY-MM-DD HH:mm:ss 标准格式
    - 多标签用英文逗号分隔
    - 返回带 UTF-8 BOM 的 CSV 文件，Excel 可直接打开
    """
    try:
        csv_content = experiment_service.export_experiments_csv(
            experiment_ids=request.experiment_ids,
        )
        # 生成文件名：带时间戳
        filename = f"experiments_export_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # Response 直接返回 CSV 内容，避免 JSON 转义
        return Response(
            content=csv_content.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出 CSV 失败: {str(e)}")


@router.get("/{experiment_id}")
async def get_experiment(
    experiment_id: str,
    detail: bool = Query(False, description="是否返回全量详情（层详情、时序数据、训练日志）"),
):
    """获取单个实验详情

    - detail=false（默认）：保持向后兼容，仅返回基础字段
    - detail=true：返回完整详情，包含 basic_info、model_config、layers、
                  hyperparams（合并）、metrics_summary、training_history、training_logs
    """
    if detail:
        result = experiment_service.get_experiment_detail(experiment_id)
    else:
        result = experiment_service.get_experiment(experiment_id)
    if not result:
        raise HTTPException(status_code=404, detail="实验不存在")
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.put("/{experiment_id}")
async def update_experiment(experiment_id: str, request: UpdateExperimentRequest):
    """更新实验"""
    update_data = request.model_dump(exclude_none=True)
    result = experiment_service.update_experiment(experiment_id, **update_data)
    if not result:
        raise HTTPException(status_code=404, detail="实验不存在")
    return {
        "code": 200,
        "message": "更新成功",
        "data": result,
    }


@router.put("/{experiment_id}/rename")
async def rename_experiment(experiment_id: str, request: RenameExperimentRequest):
    """重命名实验

    - 名称长度 2-50 字符
    - 支持中文、英文、数字、下划线、连字符、空格
    - 自动校验名称唯一性（不区分大小写）
    """
    import re
    name = request.name.strip()
    # 校验名称格式：允许中文、英文、数字、下划线、连字符、空格
    if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_\- ]+$', name):
        raise HTTPException(
            status_code=400,
            detail="实验名称仅支持中文、英文、数字、下划线、连字符和空格",
        )
    if len(name) < 2 or len(name) > 50:
        raise HTTPException(status_code=400, detail="实验名称长度需在2-50字符之间")

    result = experiment_service.rename_experiment(experiment_id, name)
    if not result:
        raise HTTPException(status_code=404, detail="实验不存在")
    if result == "duplicate":
        raise HTTPException(status_code=409, detail="实验名称已存在，请使用其他名称")
    return {
        "code": 200,
        "message": "重命名成功",
        "data": result,
    }


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """删除实验（软删除）"""
    success = experiment_service.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="实验不存在")
    return {
        "code": 200,
        "message": "删除成功",
        "data": None,
    }


@router.post("/batch/delete")
async def batch_delete_experiments(request: BatchDeleteRequest):
    """批量删除实验（软删除）

    - 支持按ID列表删除或全部删除
    - delete_all=true 时忽略 experiment_ids，删除所有未删除实验
    """
    try:
        count = experiment_service.batch_delete_experiments(
            experiment_ids=request.experiment_ids,
            delete_all=request.delete_all,
        )
        return {
            "code": 200,
            "message": f"成功删除 {count} 个实验",
            "data": {"deleted": count},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量删除失败: {str(e)}")


# ============================================================
# 自定义实验模板 API
# ============================================================

class SaveTemplateRequest(BaseModel):
    """保存自定义实验模板"""
    name: str = Field(..., min_length=2, max_length=50, description="模板名称")
    description: Optional[str] = Field("", description="模板描述")
    template_type: str = Field("comparison", description="模板类型: comparison / ablation")
    configs: List[Dict[str, Any]] = Field(..., description="实验配置列表")
    comparison_metrics: Optional[List[str]] = Field(
        ["val_acc", "val_loss", "best_accuracy"],
        description="对比指标列表",
    )


@router.get("/templates")
async def list_templates(
    template_type: Optional[str] = Query(None, description="模板类型: comparison / ablation"),
):
    """获取自定义实验模板列表"""
    result = experiment_service.list_templates(template_type=template_type)
    return {
        "code": 200,
        "message": "获取成功",
        "data": result,
    }


@router.post("/templates")
async def save_template(request: SaveTemplateRequest):
    """保存自定义实验模板"""
    try:
        result = experiment_service.save_template(
            name=request.name,
            description=request.description,
            template_type=request.template_type,
            configs=request.configs,
            comparison_metrics=request.comparison_metrics,
        )
        return {
            "code": 200,
            "message": "模板保存成功",
            "data": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存模板失败: {str(e)}")


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str):
    """删除自定义实验模板"""
    success = experiment_service.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="模板不存在")
    return {
        "code": 200,
        "message": "删除成功",
        "data": None,
    }


@router.get("/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    metric_type: Optional[str] = None,
    limit: int = Query(1000, ge=1, le=10000),
):
    """获取实验指标数据"""
    exp = experiment_service.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")

    metrics = experiment_service.get_metrics(
        experiment_id=experiment_id,
        metric_type=metric_type,
        limit=limit,
    )
    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "experiment_id": experiment_id,
            "metrics": metrics,
            "count": len(metrics),
        },
    }


@router.post("/{experiment_id}/metrics")
async def add_experiment_metrics(experiment_id: str, request: AddMetricsRequest):
    """批量添加实验指标"""
    exp = experiment_service.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")

    try:
        count = experiment_service.add_metrics(experiment_id, request.metrics)
        return {
            "code": 200,
            "message": f"成功添加 {count} 条指标",
            "data": {"added": count},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加指标失败: {str(e)}")


@router.get("/{experiment_id}/export/metrics-csv")
async def export_experiment_metrics_csv(experiment_id: str):
    """导出单个实验的逐epoch训练指标为CSV（科研用）

    包含：epoch, train_loss, val_loss, train_acc, val_acc, precision, recall, f1,
          learning_rate, gradient_norm, weight_norm
    """
    try:
        csv_content = experiment_service.export_experiment_metrics_csv(experiment_id)
        exp = experiment_service.get_experiment(experiment_id)
        safe_name = (exp.get("name", "experiment") if exp else "experiment").replace(" ", "_")
        filename = f"{safe_name}_metrics_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=csv_content.encode("utf-8"),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出指标CSV失败: {str(e)}")


@router.get("/{experiment_id}/export/json")
async def export_experiment_json(experiment_id: str):
    """导出单个实验的完整数据为JSON（包含配置、时序指标、日志、混淆矩阵）"""
    import json as _json
    try:
        data = experiment_service.export_experiment_json(experiment_id)
        filename = f"experiment_{experiment_id[:8]}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return Response(
            content=_json.dumps(data, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出JSON失败: {str(e)}")
