"""
数据集管理 API 路由
"""
import os
import uuid
import json
import logging
import threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List

from app.services.dataset_service import DatasetService
from app.ml.datasets.builtin_downloader import (
    list_builtin_datasets,
    get_builtin_dataset_info,
    prepare_mnist_dataset,
    prepare_cifar10_dataset,
)

logger = logging.getLogger(__name__)

router = APIRouter()
dataset_service = DatasetService()

_MAX_UPLOAD_SIZE = 500 * 1024 * 1024

_download_tasks = {}
_download_lock = threading.Lock()


@router.get("")
async def list_datasets(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    search: Optional[str] = None,
    tags: Optional[str] = None,
    status: Optional[str] = None,
):
    result = dataset_service.list_datasets(
        page=page, page_size=page_size, search=search, tags=tags, status=status,
    )
    return {"code": 200, "message": "获取成功", "data": result}


@router.get("/builtin")
async def list_builtin():
    """列出所有内置标准数据集"""
    datasets = list_builtin_datasets()
    return {"code": 200, "message": "获取成功", "data": datasets}


@router.get("/builtin/{name}")
async def get_builtin_info(name: str):
    """获取内置数据集详情"""
    info = get_builtin_dataset_info(name)
    if not info:
        raise HTTPException(status_code=404, detail=f"内置数据集 {name} 不存在")
    return {"code": 200, "message": "获取成功", "data": info}


@router.post("/builtin/{name}/download")
async def download_builtin_dataset(name: str):
    """下载并注册内置标准数据集（MNIST/CIFAR-10）"""
    if name not in ("mnist", "cifar10"):
        raise HTTPException(status_code=400, detail=f"不支持的内置数据集: {name}")

    from app.core.database import SessionLocal
    from app.models.dataset import Dataset as DatasetModel

    cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
    processed_file = os.path.join(cache_dir, name, f"{name}_processed.pkl")

    db = SessionLocal()
    try:
        existing = db.query(DatasetModel).filter(
            DatasetModel.file_path == f"builtin://{name}",
            DatasetModel.is_deleted == False,
            DatasetModel.status == "ready",
        ).first()
        if existing and os.path.exists(processed_file):
            return {
                "code": 200,
                "message": "数据集已存在",
                "data": {
                    "task_id": "",
                    "dataset_name": name,
                    "status": "completed",
                    "progress": 100,
                    "message": "数据集已存在，无需重新下载",
                    "dataset_id": existing.dataset_id,
                }
            }
    finally:
        db.close()

    with _download_lock:
        if name in _download_tasks and _download_tasks[name].get("status") in ("downloading", "processing", "registering"):
            return {"code": 200, "message": "下载任务已在进行中", "data": _download_tasks[name]}

        task_id = str(uuid.uuid4())
        _download_tasks[name] = {
            "task_id": task_id,
            "dataset_name": name,
            "status": "starting",
            "progress": 0,
            "message": "准备下载...",
        }

    def progress_callback(percent, downloaded, total, msg=""):
        with _download_lock:
            _download_tasks[name]["progress"] = percent
            _download_tasks[name]["message"] = msg or f"下载中 {percent}%"
            _download_tasks[name]["status"] = "downloading" if percent < 100 else "processing"

    def download_thread():
        cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with _download_lock:
                _download_tasks[name]["status"] = "downloading"
                _download_tasks[name]["message"] = "开始下载..."

            if name == "mnist":
                data = prepare_mnist_dataset(cache_dir, progress_callback=progress_callback)
                num_classes = 10
                class_names = [str(i) for i in range(10)]
                feature_shape = "1x28x28"
                img_size = (28, 28)
                ds_type = "mnist_idx"
            elif name == "cifar10":
                data = prepare_cifar10_dataset(cache_dir, progress_callback=progress_callback)
                num_classes = 10
                class_names = data.get("class_names", ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
                feature_shape = "3x32x32"
                img_size = (32, 32)
                ds_type = "image_folder"
            else:
                raise ValueError(f"未知数据集: {name}")

            with _download_lock:
                _download_tasks[name]["status"] = "registering"
                _download_tasks[name]["message"] = "注册数据集..."

            X = data["X_train"]
            y = data["y_train"]

            dataset_id = str(uuid.uuid4())
            from app.core.database import SessionLocal
            from app.models.dataset import Dataset as DatasetModel
            from datetime import datetime

            db = SessionLocal()
            try:
                class_dist = {}
                for label in y:
                    l = int(label)
                    class_dist[str(l)] = class_dist.get(str(l), 0) + 1

                ds = DatasetModel(
                    dataset_id=dataset_id,
                    name="MNIST 手写数字" if name == "mnist" else "CIFAR-10",
                    description="自动下载的标准数据集" if name == "mnist" else "自动下载的CIFAR-10彩色图像数据集",
                    version="v1",
                    file_path="builtin://" + name,
                    extract_path=os.path.join(cache_dir, name),
                    sample_count=len(y),
                    class_count=num_classes,
                    image_size=feature_shape,
                    feature_shape=feature_shape,
                    dataset_type=ds_type,
                    class_distribution=json.dumps(class_dist),
                    status="ready",
                    tags="builtin,standard" + (",mnist" if name == "mnist" else ",cifar10"),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(ds)
                db.commit()

                dataset_service._cache.pop(dataset_id, None)

                with _download_lock:
                    _download_tasks[name]["status"] = "completed"
                    _download_tasks[name]["progress"] = 100
                    _download_tasks[name]["message"] = "下载完成！"
                    _download_tasks[name]["dataset_id"] = dataset_id
                    _download_tasks[name]["sample_count"] = len(y)
                    _download_tasks[name]["class_count"] = num_classes
            finally:
                db.close()

        except Exception as e:
            logger.error(f"下载内置数据集 {name} 失败: {e}", exc_info=True)
            with _download_lock:
                _download_tasks[name]["status"] = "failed"
                _download_tasks[name]["message"] = f"下载失败: {str(e)}"

    thread = threading.Thread(target=download_thread, daemon=True)
    thread.start()

    return {"code": 200, "message": "下载任务已启动", "data": _download_tasks[name]}


@router.get("/builtin/{name}/status")
async def get_download_status(name: str):
    """查询内置数据集下载进度"""
    with _download_lock:
        task = _download_tasks.get(name, {})
    if not task:
        return {"code": 200, "data": {"status": "not_started", "progress": 0}}
    return {"code": 200, "data": task}


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, samples: int = Query(16, ge=1, le=64)):
    """数据集预览：返回样本图片base64和统计信息"""
    ds_info = dataset_service.get_dataset(dataset_id)
    if not ds_info:
        raise HTTPException(status_code=404, detail="数据集不存在")

    extract_path = ds_info.get("extract_path")
    dataset_type = ds_info.get("dataset_type", "")

    import base64
    import io
    import numpy as np
    from PIL import Image

    try:
        ds_data = dataset_service.load_dataset(dataset_id)
        X = ds_data["X"]
        y = ds_data["y"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载数据集失败: {str(e)}")

    total = len(y)
    n_samples = min(samples, total)
    indices = np.random.choice(total, n_samples, replace=False) if total > n_samples else np.arange(total)

    sample_images = []
    for idx in indices:
        img = X[idx]
        label = int(y[idx])
        try:
            if img.ndim == 2:
                pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
            elif img.ndim == 3:
                if img.shape[0] in (1, 3):
                    img_chw = img.transpose(1, 2, 0)
                else:
                    img_chw = img
                if img_chw.shape[-1] == 1:
                    pil_img = Image.fromarray(img_chw.squeeze().astype(np.uint8), mode='L')
                elif img_chw.shape[-1] == 3:
                    pil_img = Image.fromarray(img_chw.astype(np.uint8), mode='RGB')
                else:
                    pil_img = Image.fromarray(img_chw[:, :, :3].astype(np.uint8), mode='RGB')
            else:
                pil_img = Image.fromarray(img.flatten()[:784].reshape(28, 28).astype(np.uint8), mode='L')

            pil_img = pil_img.resize((64, 64), Image.NEAREST)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            sample_images.append({"index": int(idx), "label": label, "image": f"data:image/png;base64,{b64}"})
        except Exception as e:
            logger.warning(f"预览样本 {idx} 失败: {e}")

    class_dist = ds_info.get("class_distribution", {})
    stats = {
        "sample_count": total,
        "class_count": ds_info.get("class_count", 0),
        "feature_shape": ds_info.get("feature_shape", ""),
        "class_distribution": class_dist,
        "pixel_mean": float(np.mean(X[:min(1000, total)])) if X.size > 0 else 0,
        "pixel_std": float(np.std(X[:min(1000, total)])) if X.size > 0 else 0,
        "value_range": [float(np.min(X)), float(np.max(X))] if X.size > 0 else [0, 0],
    }

    return {
        "code": 200,
        "message": "获取成功",
        "data": {
            "info": ds_info,
            "stats": stats,
            "samples": sample_images,
        }
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    result = dataset_service.get_dataset(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return {"code": 200, "message": "获取成功", "data": result}


@router.get("/{name}/versions")
async def get_dataset_versions(name: str):
    versions = dataset_service.get_versions(name)
    return {"code": 200, "message": "获取成功", "data": {"name": name, "versions": versions, "count": len(versions)}}


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=128),
    description: str = Form(""),
    tags: Optional[str] = Form(None),
):
    content = await file.read()
    if len(content) > _MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {_MAX_UPLOAD_SIZE // (1024*1024)}MB）")
    filename = file.filename or "dataset.zip"
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=415, detail="仅支持 zip 格式压缩包")
    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        result = dataset_service.upload_dataset(
            file_content=content, filename=filename, name=name, description=description, tags=tag_list,
        )
        return {"code": 200, "message": "数据集上传成功，正在后台解析", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据集上传失败: {str(e)}")


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    success = dataset_service.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return {"code": 200, "message": "删除成功", "data": None}


@router.post("/upload/direct")
async def upload_direct_dataset(
    file: UploadFile = File(...),
    name: str = Form("", min_length=0, max_length=128),
    description: str = Form(""),
    tags: Optional[str] = Form(None),
):
    """直接上传非压缩格式数据集文件（CSV / JSON / NumPy）

    支持格式: .csv, .tsv, .json, .npy, .npz
    文件直接保存到解析目录，无需解压。
    """
    content = await file.read()
    if len(content) > _MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制（最大 {_MAX_UPLOAD_SIZE // (1024*1024)}MB）")

    filename = file.filename or "dataset"
    file_ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    supported = {"csv", "tsv", "json", "npy", "npz"}
    if file_ext not in supported:
        raise HTTPException(status_code=415, detail=f"不支持的文件格式: .{file_ext}，支持: {', '.join(sorted(supported))}")

    tag_list = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        result = dataset_service.upload_direct_dataset(
            file_content=content, filename=filename, name=name or filename,
            description=description, tags=tag_list,
        )
        return {"code": 200, "message": "数据集上传成功，正在后台解析", "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据集上传失败: {str(e)}")


@router.post("/{dataset_id}/reparse")
async def reparse_dataset(dataset_id: str):
    result = dataset_service.reparse_dataset(dataset_id)
    if not result:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return {"code": 200, "message": "重新解析已启动", "data": result}
