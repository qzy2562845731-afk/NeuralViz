"""
模型元数据表
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime
from app.core.database import Base

class ModelMetadata(Base):
    """模型元数据"""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(256), nullable=False)
    format = Column(String(16), nullable=False)
    file_path = Column(String(512), nullable=False)
    total_params = Column(Integer, default=0)
    layer_count = Column(Integer, default=0)
    layer_info = Column(Text, nullable=True)  # JSON
    input_shape = Column(Text, nullable=True)  # JSON
    output_shape = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelMetadata {self.name} ({self.format})>"
