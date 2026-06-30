"""
实验元数据模型
存储实验的基本信息、配置、模型结构
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, Float, Boolean
from app.core.database import Base


class Experiment(Base):
    """实验表"""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(String(1024), nullable=True, default="")

    model_id = Column(String(64), nullable=True)
    model_name = Column(String(256), nullable=True)
    model_architecture = Column(Text, nullable=True)

    status = Column(String(32), default="draft", nullable=False)

    total_params = Column(Integer, default=0)
    layer_count = Column(Integer, default=0)

    best_accuracy = Column(Float, default=0.0)
    final_loss = Column(Float, default=0.0)
    total_epochs = Column(Integer, default=0)
    current_step = Column(Integer, default=0)

    hyperparams = Column(Text, nullable=True)
    config = Column(Text, nullable=True)
    tags = Column(String(512), nullable=True, default="")

    is_deleted = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
