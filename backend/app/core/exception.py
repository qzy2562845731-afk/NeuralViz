from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """应用全局自定义异常基类"""

    def __init__(
        self,
        message: str = "服务内部异常",
        code: int = 500,
        details: Optional[dict] = None,
    ):
        self.message = message
        self.code = code
        self.details = details
        super().__init__(self.message)


class BusinessException(AppException):
    """通用业务异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=message,
            code=400,
            details=details,
        )


class NotFoundException(AppException):
    """资源不存在异常"""

    def __init__(self, resource: str, resource_id: Optional[str] = None):
        msg = f"资源不存在: {resource}"
        if resource_id:
            msg += f" ({resource_id})"
        super().__init__(
            message=msg,
            code=404,
            details={"resource": resource, "resource_id": resource_id},
        )


class PermissionDeniedException(AppException):
    """权限不足异常"""

    def __init__(self, message: str = "权限不足"):
        super().__init__(
            message=message,
            code=403,
            details=None,
        )


class ModelNotFoundException(AppException):
    """模型未找到异常"""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"模型不存在: {model_id}",
            code=404,
            details={"model_id": model_id},
        )


class ModelParseException(AppException):
    """模型解析失败异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=f"模型解析失败: {message}",
            code=422,
            details=details,
        )


class InferenceException(AppException):
    """推理失败异常"""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message=f"推理失败: {message}",
            code=422,
            details=details,
        )


class UnsupportedFormatException(AppException):
    """不支持的模型格式异常"""

    def __init__(self, format: str):
        super().__init__(
            message=f"不支持的模型格式: {format}",
            code=415,
            details={"format": format},
        )


async def app_exception_handler(request: Request, exc: AppException):
    """应用异常处理器"""
    return JSONResponse(
        status_code=exc.code,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": exc.details,
        },
    )


async def http_exception_handler(request: Request, exc: Exception):
    """HTTP 异常处理器（占位，供 FastAPI 注册）"""
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={
            "code": getattr(exc, "status_code", 500),
            "message": getattr(exc, "detail", "请求异常"),
            "data": None,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": f"服务器内部错误: {str(exc)}",
            "data": None,
        },
    )
