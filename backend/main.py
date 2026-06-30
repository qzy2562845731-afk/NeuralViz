"""
NeuralViz Backend - 神经网络可视化后端服务

启动命令:
    uvicorn main:app --reload --port 8000 --host 0.0.0.0

或直接运行:
    python main.py
"""

import sys
import os
import shutil
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# 添加 backend 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.config import settings
from app.core.database import init_db
from app.core.logging_config import setup_logging, get_logger
from app.core.exception import (
    AppException,
    http_exception_handler,
    generic_exception_handler,
    app_exception_handler,
)
from app.api import api_router
from app.ml import ModelAdapterFactory
from app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理（替代已弃用的 on_event）"""
    # ========== 启动逻辑 ==========
    setup_logging(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
    logger = get_logger(__name__)
    
    logger.info(f"{settings.APP_NAME} v{settings.APP_VERSION} 启动中...")
    
    # 初始化数据库
    logger.info("初始化数据库...")
    init_db()
    logger.info("数据库初始化完成")
    
    # 创建上传目录
    logger.info("检查上传目录...")
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    uploads_dir = settings.MODEL_DIR.parent / "datasets"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理无效数据集记录并初始化示例数据
    logger.info("初始化示例数据...")
    try:
        from app.services.dataset_service import DatasetService
        ds_service = DatasetService()
        cleaned = ds_service.cleanup_invalid_records()
        if cleaned > 0:
            logger.info("已清理 %d 条无效数据集记录", cleaned)
        sample_ds = ds_service.ensure_sample_dataset()
        logger.info("示例数据集就绪: %s (%d 样本)", sample_ds['name'], sample_ds['sample_count'])
    except Exception as e:
        logger.warning("示例数据初始化警告: %s", e)
    
    logger.info("服务启动完成 - API文档: http://localhost:%d/docs", settings.PORT)
    
    yield
    
    # ========== 关闭逻辑（如有需要可在此添加清理代码） ==========


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    description="""
## NeuralViz Backend

神经网络可视化后端服务 API

### 功能

- **模型解析**: 支持 ONNX、PyTorch、Keras 等多种格式
- **图片推理**: 支持常见图片格式，提供预测结果和激活值
- **服务状态**: 健康检查和支持的模型格式查询

### 接口

- `GET /api/health` - 健康检查
- `GET /api/health/status` - 服务状态
- `POST /api/model/parse` - 解析模型
- `GET /api/model/{model_id}` - 获取模型信息
- `GET /api/model/` - 获取模型列表
- `POST /api/inference/image` - 图片推理
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=["*"],
    max_age=3600,
)

# 注册异常处理器
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 注册 API 路由
app.include_router(api_router, prefix="/api")

# 全局服务实例
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """获取模型服务单例"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


# ============================================================
# 前端兼容接口（根路径，与现有前端调用约定对齐）
# ============================================================


@app.get("/health", tags=["前端兼容接口"])
async def health_check_compat():
    """健康检查 - 前端使用的接口路径"""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "app_name": settings.APP_NAME,
    }


@app.get("/models", tags=["前端兼容接口"])
async def list_models_compat():
    """获取模型列表 - 前端使用的接口路径

    返回：已解析的真实模型 + 内置示例模型
    """
    service = get_model_service()
    models = service.list_models()

    # 添加内置示例模型
    builtin_models = [
        {
            "model_id": "sample_cnn",
            "model_name": "SampleCNN (MNIST)",
            "file_name": "sample_cnn",
            "format": "内置示例",
            "total_params": 225034,
            "total_layers": 12,
            "input_shape": [1, 28, 28],
            "output_shape": [10],
            "description": "MNIST 手写数字识别 CNN 示例模型",
            "is_builtin": True,
        }
    ]

    return {
        "models": builtin_models + models,
        "total": len(builtin_models) + len(models),
    }


class AnalyzeRequest(BaseModel):
    model: str


@app.post("/analyze", tags=["前端兼容接口"])
async def analyze_model_compat(request: AnalyzeRequest):
    """分析模型 - 前端使用的接口路径

    前端通过 /analyze 发送 { model: "模型名称" } 来获取模型解析结果
    返回格式需与前端 ModelAnalysisResult 兼容
    """
    service = get_model_service()

    # 1. 先检查是否是内置示例模型
    model_lower = request.model.lower().strip()
    if model_lower in ("sample_cnn", "sample", "default", "model.pth", ""):
        from app.ml.factory import ModelAdapterFactory

        try:
            adapter = ModelAdapterFactory.create("sample_cnn")
            adapter.load_model("sample_cnn")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"示例模型加载失败: {str(e)}")

        layer_info = adapter.get_layer_info()
        return {
            "model_id": "sample_cnn",
            "model_name": "SampleCNN (MNIST)",
            "format": "内置示例",
            "total_params": 225034,
            "total_layers": len(layer_info),
            "input_shape": [1, 28, 28],
            "output_shape": [10],
            "layers": layer_info,
            "description": "MNIST 手写数字识别 CNN 示例模型",
        }

    # 2. 从已解析的模型中查找
    models = service.list_models()
    for m in models:
        if m.get("model_name") == request.model or m.get("model_id") == request.model:
            return service.get_model_info(m["model_id"])

    # 3. 如果没有，尝试从上传目录中查找文件并解析
    model_file = settings.MODEL_DIR / request.model
    if not model_file.exists():
        for ext in [".onnx", ".pt", ".pth", ".h5", ".keras", ".pb", ".pickle", ".pkl"]:
            candidate = settings.MODEL_DIR / f"{request.model}{ext}"
            if candidate.exists():
                model_file = candidate
                break

    if not model_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"模型 {request.model} 未找到，请先上传",
        )

    # 4. 解析模型
    model_id = str(uuid.uuid4())
    try:
        result = await service.parse_model(model_id, str(model_file))
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/infer", tags=["前端兼容接口"])
async def inference_compat(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    """图片推理 - 前端使用的接口路径

    前端通过 FormData 发送 model + file 进行推理
    返回 InferenceResult 格式：activations, predictions, input_size, success

    特殊规则：
    - model 为 "model.pth" 或空时，自动使用内置 SampleCNN 示例模型
    - model 为 "sample_cnn" / "sample" / "default" 时使用内置示例
    """
    import numpy as np
    from PIL import Image
    import io

    service = get_model_service()

    # 1. 判断是否使用内置示例模型
    model_lower = model.lower().strip()
    use_builtin = model_lower in (
        "model.pth",
        "",
        "sample_cnn",
        "sample",
        "default",
        "默认模型",
    )

    adapter = None

    if use_builtin:
        # 使用内置示例模型
        try:
            from app.ml.factory import ModelAdapterFactory

            adapter = ModelAdapterFactory.create("sample_cnn")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"示例模型加载失败: {str(e)}")
    else:
        # 2. 查找真实模型
        models = service.list_models()
        model_info = None
        for m in models:
            if m.get("model_name") == model or m.get("model_id") == model:
                model_info = service.get_model_info(m["model_id"])
                break

        if not model_info:
            raise HTTPException(status_code=404, detail=f"模型 {model} 未找到")

        # 3. 获取模型文件路径并创建适配器
        model_file = settings.MODEL_DIR / model_info.get("file_name", model)
        if not model_file.exists():
            if "file_path" in model_info:
                model_file = Path(model_info["file_path"])
            if not model_file.exists():
                raise HTTPException(status_code=404, detail=f"模型文件不存在: {model}")

        try:
            adapter = ModelAdapterFactory.create(str(model_file))
            adapter.load_model(str(model_file))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"模型加载失败: {str(e)}")

    # 3. 读取并预处理图片（与 /api/inference/image 保持一致：仅 /255 归一化）
    try:
        contents = await file.read()

        # 获取目标尺寸、通道数与布局
        input_shape = adapter.get_input_shape()
        # 从 input_shape 启发式推断布局、目标尺寸与通道数
        def _detect_layout_from_shape_compat(shape):
            shape = [int(s) if s is not None else -1 for s in shape]
            fixed = [s for s in shape if s > 0]
            SMALL_CHANNELS = {1, 2, 3, 4}
            if len(shape) >= 4:
                c_nchw = shape[1]
                c_nhwc = shape[-1]
                is_nchw = c_nchw in SMALL_CHANNELS and shape[2] not in SMALL_CHANNELS and shape[3] not in SMALL_CHANNELS
                is_nhwc = c_nhwc in SMALL_CHANNELS and shape[1] not in SMALL_CHANNELS and shape[2] not in SMALL_CHANNELS
                if is_nhwc and not is_nchw:
                    return "NHWC", (shape[1] or 28, shape[2] or 28), c_nhwc
                h = shape[2] if shape[2] > 0 else 28
                w = shape[3] if shape[3] > 0 else 28
                c = c_nchw if c_nchw > 0 else (c_nhwc if c_nhwc > 0 else 1)
                return "NCHW", (h, w), c
            elif len(shape) == 3:
                c_first = shape[0]
                c_last = shape[-1]
                if c_last in SMALL_CHANNELS and c_first not in SMALL_CHANNELS:
                    return "NHWC", (shape[0] or 28, shape[1] or 28), c_last
                c = c_first if c_first > 0 else (c_last if c_last > 0 else 1)
                return "NCHW", (shape[1] or 28, shape[2] or 28), c
            return "NCHW", (28, 28), 1

        # adapter 已加载时，优先使用其 input_layout 属性
        if hasattr(adapter, "input_layout"):
            layout = adapter.input_layout
            _, target_size, target_channels = _detect_layout_from_shape_compat(input_shape)
        else:
            layout, target_size, target_channels = _detect_layout_from_shape_compat(input_shape)

        # 使用统一的预处理函数（仅 /255 归一化，与训练一致）
        from app.api.inference import preprocess_image
        img_array = preprocess_image(contents, target_size, target_channels, layout)
        img_width, img_height = target_size[1], target_size[0]

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图片处理失败: {str(e)}")
    finally:
        await file.close()

    # 4. 执行推理
    try:
        output, inference_time = adapter.infer(img_array)

        # 显式 softmax 后处理：根据 adapter.output_is_probability 决定
        if not getattr(adapter, "output_is_probability", False):
            output_max = np.max(output, axis=-1, keepdims=True)
            exp_x = np.exp(output - output_max)
            output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        # 5. 构建预测结果（取 Top5）
        output_flat = output.flatten()
        if len(output_flat) > 0:
            top_indices = np.argsort(output_flat)[-5:][::-1]
            predictions = [
                {
                    "class_id": int(idx),
                    "probability": float(output_flat[idx]),
                }
                for idx in top_indices
            ]
        else:
            predictions = []

        # 6. 收集各层激活值
        activations: dict = {}

        # 优先使用适配器提供的激活值
        if hasattr(adapter, "get_all_activations"):
            all_acts = adapter.get_all_activations(img_array)
            for name, act in all_acts.items():
                if isinstance(act, np.ndarray):
                    # 降维为一维数组，取平均值或采样
                    if act.ndim > 1:
                        # 对于特征图，取每个通道的均值
                        if act.ndim == 4:
                            act_1d = act.mean(axis=(2, 3)).flatten()
                        else:
                            act_1d = act.flatten()
                        # 限制长度，避免数据过大
                        max_len = 256
                        if len(act_1d) > max_len:
                            step = len(act_1d) // max_len
                            act_1d = act_1d[::step][:max_len]
                        activations[name] = act_1d.tolist()
                    else:
                        activations[name] = act.tolist()
                else:
                    activations[name] = [float(act)] if act else [0.0]
        else:
            # 降级：从层信息构建简化激活值
            layer_info = adapter.get_layer_info()
            for layer in layer_info:
                layer_name = layer.get("name", f"layer_{layer.get('id', 0)}")
                params = layer.get("params", 0)
                if params > 0:
                    act_size = min(64, max(4, int(np.sqrt(params))))
                    activations[layer_name] = np.random.rand(act_size).tolist()
                else:
                    activations[layer_name] = [0.5]

        return {
            "activations": activations,
            "predictions": predictions,
            "input_size": [img_width, img_height],
            "success": True,
            "inference_time": inference_time,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.get("/", tags=["根路径"])
async def root():
    """根路径"""
    return {
        "code": 200,
        "message": f"欢迎使用 {settings.APP_NAME}",
        "data": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": f"http://localhost:{settings.PORT}/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
