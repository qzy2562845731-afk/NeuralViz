"""
实验指标模型
存储实验的训练指标数据（loss、accuracy 等）
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey
from app.core.database import Base


class ExperimentMetric(Base):
    """实验指标表"""
    __tablename__ = "experiment_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), index=True, nullable=False)

    step = Column(Integer, default=0, nullable=False)
    epoch = Column(Integer, default=0)

    loss = Column(Float, default=0.0)
    accuracy = Column(Float, default=0.0)
    val_loss = Column(Float, default=0.0)
    val_accuracy = Column(Float, default=0.0)

    learning_rate = Column(Float, default=0.0)
    batch_size = Column(Integer, default=0)

    metric_type = Column(String(32), default="training", nullable=False)
    extra_data = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
