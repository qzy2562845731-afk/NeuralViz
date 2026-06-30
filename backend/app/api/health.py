from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("")
async def health_check():
    """
    健康检查接口
    """
    return {
        "code": 200,
        "message": "服务运行正常",
        "data": {
            "status": "online",
            "version": settings.APP_VERSION,
            "app_name": settings.APP_NAME
        }
    }

@router.get("/status")
async def service_status():
    """
    服务状态与支持的模型格式列表
    """
    return {
        "code": 200,
        "message": "服务正常",
        "data": {
            "status": "online",
            "version": settings.APP_VERSION,
            "supported_formats": settings.SUPPORTED_MODEL_FORMATS
        }
    }
