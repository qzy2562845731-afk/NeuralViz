import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """应用配置"""
    APP_NAME: str = "NeuralViz Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 配置（开发环境完全开放，便于前后端联调）
    # 注意：生产环境请修改为具体的前端域名
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    # 上传目录
    UPLOAD_DIR: Path = Path(__file__).parent.parent.parent / "uploads"
    MODEL_DIR: Path = UPLOAD_DIR / "models"
    IMAGE_DIR: Path = UPLOAD_DIR / "images"
    DATASET_DIR: Path = UPLOAD_DIR / "datasets"
    
    # 数据库
    DATABASE_URL: str = "sqlite:///./neuralviz.db"
    
    # 支持的模型格式
    SUPPORTED_MODEL_FORMATS: list[dict] = [
        {"ext": ".onnx", "name": "ONNX", "adapter": "onnx"},
        {"ext": ".pt", "name": "PyTorch", "adapter": "pytorch"},
        {"ext": ".pth", "name": "PyTorch State Dict", "adapter": "pytorch"},
        {"ext": ".h5", "name": "Keras", "adapter": "pytorch"},
        {"ext": ".hdf5", "name": "Keras HDF5", "adapter": "pytorch"},
        {"ext": ".pb", "name": "TensorFlow SavedModel", "adapter": "onnx"},
        {"ext": ".keras", "name": "Keras", "adapter": "pytorch"},
        {"ext": ".json", "name": "JSON Config", "adapter": "onnx"},
        {"ext": ".js", "name": "JavaScript Config", "adapter": "onnx"},
        {"ext": ".pickle", "name": "Pickle", "adapter": "pytorch"},
        {"ext": ".pkl", "name": "Pickle", "adapter": "pytorch"},
    ]
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

# 确保上传目录存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
