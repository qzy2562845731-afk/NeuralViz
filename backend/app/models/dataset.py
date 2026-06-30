"""
数据集元数据模型
存储数据集的基本信息、文件路径、解析统计结果
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean
from app.core.database import Base


class Dataset(Base):
    """数据集表"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True, default="")
    version = Column(String(32), nullable=False, default="v1")
    file_path = Column(String(256), nullable=False)
    extract_path = Column(String(256), nullable=True)
    sample_count = Column(Integer, default=0)
    class_count = Column(Integer, default=0)
    image_size = Column(String(32), nullable=True)
    class_distribution = Column(Text, nullable=True)  # JSON
    file_hash = Column(String(64), nullable=True, index=True)
    status = Column(String(16), default="uploading", nullable=False)
    error_message = Column(Text, nullable=True)
    dataset_type = Column(String(32), nullable=True)  # 格式标识：mnist_idx / numpy / csv / image_folder
    feature_shape = Column(String(64), nullable=True)  # 特征维度/图像尺寸
    tags = Column(String(256), nullable=True, default="")
    is_deleted = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Dataset {self.name} ({self.version})>"
