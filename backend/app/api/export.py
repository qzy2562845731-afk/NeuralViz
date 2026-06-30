"""
导出 API 路由
支持多格式导出：
- Excel (.xlsx): 实验数据、训练指标、混淆矩阵多sheet
- CSV: 混淆矩阵、ROC曲线数据
- JSON: 完整指标、混淆矩阵
- PNG: 高分辨率300DPI图表（训练曲线、混淆矩阵、ROC、PR曲线）
- SVG: 矢量图（论文级）
- PDF: 矢量PDF（论文级）
- ONNX: 训练好的模型导出为ONNX格式（跨平台部署）
"""
import io
import json
import logging

from app.core.security import sanitize_path_id
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.config import settings
from app.services.experiment_service import ExperimentService
from app.services.export_service import (
    export_experiments_to_excel,
    export_single_experiment_excel,
    export_confusion_matrix_csv,
    export_confusion_matrix_json,
    export_roc_data,
    export_metrics_json,
    export_chart_png,
    export_chart_svg,
    export_chart_pdf,
    export_pytorch_to_onnx,
)

logger = logging.getLogger(__name__)

router = APIRouter()
exp_service = ExperimentService()


def _get_experiment_metrics(experiment_id: str):
    """获取实验及其指标数据"""
    exp = exp_service.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")
    metrics = exp_service.get_metrics(experiment_id, limit=10000)
    return exp, metrics


def _extract_last_metric_data(metrics: List[dict]):
    """从最后一个metric中提取混淆矩阵、ROC、PR数据"""
    if not metrics:
        return None, None, None, None, None

    last_m = metrics[-1]
    extra = last_m.get("extra_data") or {}
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except:
            extra = {}

    cm = extra.get("confusion_matrix")
    roc_data = extra.get("roc_curve")
    pr_data = extra.get("pr_curve")

    class_names = None
    per_class_p = extra.get("per_class_precision", [])
    if per_class_p:
        class_names = [f"Class_{i}" for i in range(len(per_class_p))]

    return cm, roc_data, pr_data, class_names, extra


@router.get("/experiments/excel")
async def export_all_experiments_excel():
    """导出所有实验列表为Excel（多sheet）"""
    result = exp_service.list_experiments(page=1, page_size=500)
    exp_list = result.get("items", []) if isinstance(result, dict) else result

    metrics_map = {}
    for exp in exp_list:
        exp_id = exp.get("experiment_id", "")
        if exp_id:
            try:
                metrics_map[exp_id] = exp_service.get_metrics(exp_id, limit=10000)
            except:
                pass

    excel_bytes = export_experiments_to_excel(exp_list, metrics_map)

    filename = f"experiments_export_{_timestamp()}.xlsx"
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/experiment/{experiment_id}/excel")
async def export_experiment_excel(experiment_id: str):
    """导出单个实验完整报告为Excel"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    excel_bytes = export_single_experiment_excel(exp, metrics)

    filename = f"experiment_{experiment_id[:8]}_{_timestamp()}.xlsx"
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/experiment/{experiment_id}/metrics/json")
async def export_experiment_metrics_json(experiment_id: str):
    """导出实验完整指标为JSON（含ROC/PR曲线）"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, _, _ = _extract_last_metric_data(metrics)

    json_str = export_metrics_json(metrics, roc_data, pr_data)
    json_bytes = json_str.encode("utf-8")

    filename = f"metrics_{experiment_id[:8]}_{_timestamp()}.json"
    return StreamingResponse(
        io.BytesIO(json_bytes),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/experiment/{experiment_id}/confusion_matrix/{fmt}")
async def export_confusion_matrix(experiment_id: str, fmt: str, normalize: bool = Query(False)):
    """导出混淆矩阵，支持 csv / json / png / svg / pdf 格式"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, class_names, extra = _extract_last_metric_data(metrics)

    if not cm:
        raise HTTPException(status_code=400, detail="该实验无混淆矩阵数据（训练可能未完成）")

    fmt = fmt.lower()

    if fmt == "csv":
        csv_str = export_confusion_matrix_csv(cm, class_names)
        return StreamingResponse(
            io.BytesIO(csv_str.encode("utf-8-sig")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=confusion_matrix_{_timestamp()}.csv"},
        )

    elif fmt == "json":
        json_str = export_confusion_matrix_json(cm, class_names)
        return StreamingResponse(
            io.BytesIO(json_str.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=confusion_matrix_{_timestamp()}.json"},
        )

    elif fmt in ("png", "svg", "pdf"):
        chart_data = {"cm": cm, "class_names": class_names, "normalize": normalize}
        dpi = 300

        if fmt == "png":
            content = export_chart_png("confusion_matrix", chart_data, dpi=dpi)
            media_type = "image/png"
            ext = "png"
        elif fmt == "svg":
            content = export_chart_svg("confusion_matrix", chart_data)
            media_type = "image/svg+xml"
            ext = "svg"
        else:
            content = export_chart_pdf("confusion_matrix", chart_data, dpi=dpi)
            media_type = "application/pdf"
            ext = "pdf"

        filename = f"confusion_matrix_{_timestamp()}.{ext}"
        return StreamingResponse(
            io.BytesIO(content),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    else:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {fmt}，支持 csv/json/png/svg/pdf")


@router.get("/experiment/{experiment_id}/chart/{chart_type}/{fmt}")
async def export_chart(
    experiment_id: str,
    chart_type: str,
    fmt: str,
    dpi: int = Query(300, ge=72, le=600),
    normalize: bool = Query(False),
):
    """导出训练相关图表

    Args:
        chart_type: loss_curve / acc_curve / roc / pr
        fmt: png / svg / pdf
        dpi: 分辨率（默认300DPI论文级）
        normalize: 混淆矩阵是否归一化
    """
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, class_names, _ = _extract_last_metric_data(metrics)

    chart_type = chart_type.lower()
    fmt = fmt.lower()

    supported_charts = {"loss_curve", "acc_curve", "roc", "pr", "confusion_matrix"}
    if chart_type not in supported_charts:
        raise HTTPException(status_code=400, detail=f"不支持的图表类型: {chart_type}，支持 {supported_charts}")

    if fmt not in ("png", "svg", "pdf"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {fmt}，支持 png/svg/pdf")

    if chart_type in ("roc", "pr") and not ((chart_type == "roc" and roc_data) or (chart_type == "pr" and pr_data)):
        raise HTTPException(status_code=400, detail=f"该实验无{chart_type.upper()}曲线数据（训练可能未完成或类别数不足）")

    if chart_type == "confusion_matrix" and not cm:
        raise HTTPException(status_code=400, detail="该实验无混淆矩阵数据")

    chart_data = {
        "metrics": metrics,
        "cm": cm,
        "roc_data": roc_data,
        "pr_data": pr_data,
        "class_names": class_names,
        "normalize": normalize,
    }

    if fmt == "png":
        content = export_chart_png(chart_type, chart_data, dpi=dpi)
        media_type = "image/png"
        ext = "png"
    elif fmt == "svg":
        content = export_chart_svg(chart_type, chart_data)
        media_type = "image/svg+xml"
        ext = "svg"
    else:
        content = export_chart_pdf(chart_type, chart_data, dpi=dpi)
        media_type = "application/pdf"
        ext = "pdf"

    filename = f"{chart_type}_{_timestamp()}.{ext}"
    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/experiment/{experiment_id}/roc/csv")
async def export_roc_csv(experiment_id: str):
    """导出ROC曲线数据为CSV"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, class_names, _ = _extract_last_metric_data(metrics)

    if not roc_data:
        raise HTTPException(status_code=400, detail="该实验无ROC曲线数据")

    import csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([f"# Macro AUC = {roc_data['macro_auc']:.4f}"])

    for cls_name, fpr_list in roc_data["fpr"].items():
        writer.writerow([f"# Class: {cls_name}, AUC = {roc_data['auc_scores'][cls_name]:.4f}"])
        writer.writerow(["FPR", "TPR"])
        tpr_list = roc_data["tpr"][cls_name]
        for i in range(len(fpr_list)):
            writer.writerow([fpr_list[i], tpr_list[i]])
        writer.writerow([])

    output.seek(0)
    filename = f"roc_data_{_timestamp()}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/experiment/{experiment_id}/roc")
async def get_roc_data(experiment_id: str):
    """获取ROC曲线数据（JSON，用于前端绘制）"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, class_names, _ = _extract_last_metric_data(metrics)

    if not roc_data:
        return {"code": 200, "message": "无ROC数据", "data": None}

    return {"code": 200, "message": "获取成功", "data": roc_data}


@router.get("/experiment/{experiment_id}/pr")
async def get_pr_data(experiment_id: str):
    """获取PR曲线数据（JSON，用于前端绘制）"""
    exp, metrics = _get_experiment_metrics(experiment_id)
    cm, roc_data, pr_data, class_names, _ = _extract_last_metric_data(metrics)

    if not pr_data:
        return {"code": 200, "message": "无PR数据", "data": None}

    return {"code": 200, "message": "获取成功", "data": pr_data}


@router.get("/experiment/{experiment_id}/onnx")
async def export_experiment_onnx(experiment_id: str):
    """导出训练好的模型为ONNX格式

    支持跨平台推理部署，可在Netron、ONNX Runtime等工具中查看和使用
    """
    experiment_id = sanitize_path_id(experiment_id)
    exp = exp_service.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")

    model_dir = settings.MODEL_DIR / experiment_id
    if not model_dir.exists():
        raise HTTPException(status_code=400, detail="该实验无模型文件（训练可能未完成）")

    try:
        feature_shape = None
        try:
            from app.services.dataset_service import DatasetService
            ds_service = DatasetService()
            if exp.get("dataset_id"):
                ds = ds_service.get_dataset(exp["dataset_id"])
                if ds and ds.get("feature_shape"):
                    feature_shape = ds["feature_shape"]
        except Exception:
            pass

        if feature_shape:
            parts = [int(x) for x in str(feature_shape).lower().replace('c', '').split('x') if x.strip().isdigit()]
            if len(parts) == 3:
                c, h, w = parts
                input_shape = (1, c, h, w)
            else:
                input_shape = (1, 1, 28, 28)
        else:
            input_shape = (1, 1, 28, 28)

        onnx_bytes = export_pytorch_to_onnx(str(model_dir), input_shape=input_shape)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("ONNX导出失败")
        raise HTTPException(status_code=500, detail=f"ONNX导出失败: {str(e)}")

    filename = f"model_{experiment_id[:8]}_{_timestamp()}.onnx"
    return StreamingResponse(
        io.BytesIO(onnx_bytes),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _timestamp() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")
