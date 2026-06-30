"""
API 路由模块
"""
from fastapi import APIRouter
from .health import router as health_router
from .model import router as model_router
from .inference import router as inference_router
from .experiment import router as experiment_router
from .dataset import router as dataset_router
from .training import router as training_router
from .export import router as export_router
from .gradcam import router as gradcam_router

# 主 API 路由
api_router = APIRouter()

# 注册子路由
api_router.include_router(health_router, prefix="/health", tags=["健康检查"])
api_router.include_router(model_router, prefix="/model", tags=["模型管理"])
api_router.include_router(inference_router, prefix="/inference", tags=["推理服务"])
api_router.include_router(experiment_router, prefix="/experiment", tags=["实验管理"])
api_router.include_router(dataset_router, prefix="/dataset", tags=["数据集管理"])
api_router.include_router(training_router, prefix="/training", tags=["训练引擎"])
api_router.include_router(export_router, prefix="/export", tags=["数据导出"])
api_router.include_router(gradcam_router, prefix="/gradcam", tags=["Grad-CAM可视化"])
