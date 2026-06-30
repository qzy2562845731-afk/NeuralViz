import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import numpy as np

from app.core.config import settings
from app.core.exception import ModelParseException, UnsupportedFormatException
from app.ml import ModelAdapterFactory
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.post("/parse")
async def parse_model(file: UploadFile = File(...)):
    """
    解析模型文件
    
    上传模型文件，自动识别格式并解析，返回模型基本信息、层列表详情
    """
    # 生成唯一ID
    model_id = str(uuid.uuid4())
    
    # 保存上传文件
    file_ext = Path(file.filename).suffix.lower()
    model_filename = f"{model_id}{file_ext}"
    model_path = settings.MODEL_DIR / model_filename
    
    try:
        # 保存文件
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 检查格式支持
        if not ModelAdapterFactory.is_supported(str(model_path)):
            raise UnsupportedFormatException(file_ext)
        
        # 解析模型
        result = await model_service.parse_model(model_id, str(model_path))
        
        return {
            "code": 200,
            "message": "模型解析成功",
            "data": result
        }
        
    except UnsupportedFormatException as e:
        raise HTTPException(status_code=415, detail=str(e))
    except ModelParseException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型解析失败: {str(e)}")
    finally:
        await file.close()

@router.get("/{model_id}")
async def get_model_info(model_id: str):
    """
    获取模型详情
    """
    result = model_service.get_model_info(model_id)
    if not result:
        raise HTTPException(status_code=404, detail="模型不存在")
    
    return {
        "code": 200,
        "message": "获取成功",
        "data": result
    }

@router.get("/")
async def list_models():
    """
    获取已解析模型列表
    """
    models = model_service.list_models()
    return {
        "code": 200,
        "message": "获取成功",
        "data": models
    }
