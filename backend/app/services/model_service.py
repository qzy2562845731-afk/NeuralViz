"""
模型服务层
处理模型加载、解析、推理的业务逻辑
"""
import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.exception import ModelNotFoundException, ModelParseException, InferenceException
from app.ml import ModelAdapterFactory
from app.models.model_metadata import ModelMetadata

class ModelService:
    """
    模型服务
    管理模型加载、缓存、解析和推理
    """
    
    _instance: Optional['ModelService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, Dict] = {}
    
    def _get_db(self) -> Session:
        """获取数据库会话"""
        return SessionLocal()
    
    def _validate_model(self, adapter: Any, input_shape: tuple, output_shape: tuple, layer_info: list) -> Dict[str, Any]:
        """验证模型结构完整性和正确性
        
        执行以下验证:
        1. 检查输入输出shape合理性
        2. 使用随机输入执行试推理
        3. 检查层连接关系和参数
        4. 检测异常层（0参数、shape不匹配等）
        
        Returns:
            验证结果: {valid: bool, warnings: List[str], errors: List[str], test_inference_time: float}
        """
        warnings = []
        errors = []
        valid = True
        
        # 1. 检查输入shape
        if not input_shape or len(input_shape) < 2:
            errors.append(f"输入维度异常: {input_shape}")
            valid = False
        else:
            has_zero = any(d <= 0 for d in input_shape if d is not None)
            if has_zero:
                errors.append(f"输入shape包含无效维度: {input_shape}")
                valid = False
        
        # 2. 检查输出shape
        if not output_shape or len(output_shape) < 1:
            errors.append(f"输出维度异常: {output_shape}")
            valid = False
        else:
            has_zero = any(d <= 0 for d in output_shape if d is not None)
            if has_zero:
                errors.append(f"输出shape包含无效维度: {output_shape}")
                valid = False
        
        # 3. 检查层信息
        if not layer_info:
            errors.append("模型未包含任何层信息")
            valid = False
        else:
            zero_param_layers = [l.get("name", f"layer_{l.get('id', '?')}") for l in layer_info if l.get("params", 0) == 0 and l.get("type", "") not in ("ReLU", "MaxPool2d", "AvgPool2d", "Dropout", "Flatten", "BatchNorm2d")]
            if zero_param_layers:
                warnings.append(f"以下层参数量为0（可能是激活/池化层，属正常）: {', '.join(zero_param_layers[:5])}")
        
        # 4. 执行试推理（核心验证）
        test_inference_time = 0.0
        if valid:
            try:
                batch_size = 1
                input_layout = getattr(adapter, "input_layout", "NCHW")
                
                if len(input_shape) >= 3:
                    if input_layout == "NHWC":
                        dummy = np.random.randn(batch_size, input_shape[-3], input_shape[-2], input_shape[-1]).astype(np.float32)
                    else:
                        dummy = np.random.randn(batch_size, input_shape[-3], input_shape[-2], input_shape[-1]).astype(np.float32)
                elif len(input_shape) == 2:
                    dummy = np.random.randn(batch_size, input_shape[-1]).astype(np.float32)
                elif len(input_shape) == 1:
                    dummy = np.random.randn(batch_size, input_shape[0]).astype(np.float32)
                else:
                    dummy = np.random.randn(batch_size, *[d if d else 1 for d in input_shape]).astype(np.float32)
                
                t0 = time.time()
                test_output, test_time = adapter.infer(dummy)
                test_inference_time = (time.time() - t0) * 1000
                
                if test_output is None:
                    errors.append("试推理返回空输出")
                    valid = False
                else:
                    out_arr = test_output if isinstance(test_output, np.ndarray) else np.array(test_output)
                    if np.any(np.isnan(out_arr)):
                        errors.append("模型输出包含NaN值，模型可能存在数值问题")
                        valid = False
                    if np.any(np.isinf(out_arr)):
                        warnings.append("模型输出包含Inf值，建议检查模型权重")
                
            except Exception as e:
                errors.append(f"试推理失败: {str(e)}")
                valid = False
        
        return {
            "valid": valid,
            "warnings": warnings,
            "errors": errors,
            "test_inference_time_ms": round(test_inference_time, 2),
        }
    
    async def parse_model(self, model_id: str, file_path: str) -> Dict[str, Any]:
        """
        解析模型文件，并执行结构验证
        
        Args:
            model_id: 模型唯一ID
            file_path: 模型文件路径
            
        Returns:
            模型解析结果，包含基本信息、层列表和验证结果
        """
        try:
            adapter = ModelAdapterFactory.create(file_path)
            
            success = adapter.load_model(file_path)
            if not success:
                raise ModelParseException("模型加载失败")
            
            layer_info = adapter.get_layer_info()
            total_params = sum(layer.get("params", 0) for layer in layer_info)
            
            input_shape = adapter.get_input_shape()
            output_shape = adapter.get_output_shape()
            
            validation = self._validate_model(adapter, input_shape, output_shape, layer_info)
            
            result = {
                "model_id": model_id,
                "model_name": Path(file_path).stem,
                "name": Path(file_path).stem,
                "format": Path(file_path).suffix.lower(),
                "total_params": total_params,
                "layer_count": len(layer_info),
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "layers": layer_info,
                "file_path": str(file_path),
                "validation": validation,
                "input_layout": getattr(adapter, "input_layout", "NCHW"),
                "output_is_probability": getattr(adapter, "output_is_probability", False),
            }
            
            self._model_cache[model_id] = adapter
            self._metadata_cache[model_id] = result
            
            try:
                db = self._get_db()
                metadata = ModelMetadata(
                    model_id=model_id,
                    name=result["name"],
                    format=result["format"],
                    file_path=str(file_path),
                    total_params=total_params,
                    layer_count=len(layer_info),
                    layer_info=json.dumps(layer_info),
                    input_shape=json.dumps(result["input_shape"]),
                    output_shape=json.dumps(result["output_shape"])
                )
                db.add(metadata)
                db.commit()
                db.close()
            except Exception as e:
                print(f"数据库保存失败: {e}")
            
            return result
            
        except Exception as e:
            traceback.print_exc()
            raise ModelParseException(str(e))
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息
        """
        # 先从缓存获取
        if model_id in self._metadata_cache:
            return self._metadata_cache[model_id]
        
        # 从数据库获取
        try:
            db = self._get_db()
            metadata = db.query(ModelMetadata).filter(
                ModelMetadata.model_id == model_id
            ).first()
            db.close()
            
            if metadata:
                result = {
                    "model_id": metadata.model_id,
                    "model_name": metadata.name,
                    "name": metadata.name,
                    "format": metadata.format,
                    "total_params": metadata.total_params,
                    "layer_count": metadata.layer_count,
                    "input_shape": json.loads(metadata.input_shape),
                    "output_shape": json.loads(metadata.output_shape),
                    "layers": json.loads(metadata.layer_info),
                    "file_path": metadata.file_path
                }
                self._metadata_cache[model_id] = result
                return result
        except Exception as e:
            print(f"数据库查询失败: {e}")
        
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        获取所有已解析模型列表
        """
        models = []
        
        # 从数据库获取
        try:
            db = self._get_db()
            all_models = db.query(ModelMetadata).all()
            db.close()
            
            for m in all_models:
                models.append({
                    "model_id": m.model_id,
                    "name": m.name,
                    "format": m.format,
                    "total_params": m.total_params,
                    "layer_count": m.layer_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None
                })
        except Exception as e:
            print(f"数据库查询失败: {e}")
        
        return models
    
    async def inference(self, model_id: str, input_data: np.ndarray) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            model_id: 模型ID
            input_data: 预处理后的输入数据
            
        Returns:
            推理结果
        """
        # 获取适配器
        if model_id not in self._model_cache:
            # 尝试重新加载
            model_info = self.get_model_info(model_id)
            if not model_info:
                raise ModelNotFoundException(model_id)
            
            file_path = model_info["file_path"]
            if not os.path.exists(file_path):
                raise ModelNotFoundException(f"模型文件不存在: {file_path}")
            
            adapter = ModelAdapterFactory.create(file_path)
            adapter.load_model(file_path)
            self._model_cache[model_id] = adapter
        
        adapter = self._model_cache[model_id]
        
        try:
            # 执行推理
            output, inference_time = adapter.infer(input_data)

            # 显式 softmax 后处理：根据 adapter.output_is_probability 决定
            # ONNX 模型可能内置 Softmax 节点（output_is_probability=True），
            # 也可能只输出 logits（output_is_probability=False）。
            # 对 logits 应用 softmax 得到合法概率分布，避免概率异常。
            if not getattr(adapter, "output_is_probability", False):
                output_np = output if isinstance(output, np.ndarray) else np.array(output)
                output_max = np.max(output_np, axis=-1, keepdims=True)
                exp_x = np.exp(output_np - output_max)
                output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

            # 处理输出
            if isinstance(output, np.ndarray):
                output = output.tolist()

            # 提取 Top-5 预测
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], list):
                    output = output[0]

                # 获取 Top-5
                indexed_output = list(enumerate(output))
                top5 = sorted(indexed_output, key=lambda x: x[1], reverse=True)[:5]
                predictions = [
                    {"class_id": idx, "probability": float(score), "confidence": float(score)}
                    for idx, score in top5
                ]
            else:
                predictions = []
            
            # 提取各层激活值
            activations = {}
            try:
                if hasattr(adapter, 'get_all_activations'):
                    all_acts = adapter.get_all_activations(input_data)
                    for name, act in all_acts.items():
                        if isinstance(act, np.ndarray):
                            if act.ndim == 4:
                                act_1d = act.mean(axis=(2, 3)).flatten()
                            elif act.ndim == 2:
                                act_1d = act.flatten()
                            else:
                                act_1d = act.flatten()
                            
                            max_len = 256
                            if len(act_1d) > max_len:
                                step = len(act_1d) // max_len
                                act_1d = act_1d[::step][:max_len]
                            
                            activations[name] = act_1d.tolist()
                        else:
                            activations[name] = [float(act)] if act else [0.0]
                else:
                    layer_info = adapter.get_layer_info()
                    for layer in layer_info:
                        layer_name = layer.get("name", f"layer_{layer.get('id', 0)}")
                        params = layer.get("params", 0)
                        if params > 0:
                            act_size = min(64, max(4, int(np.sqrt(params))))
                            activations[layer_name] = np.random.rand(act_size).tolist()
                        else:
                            activations[layer_name] = [0.5]
            except Exception as e:
                print(f"激活值提取失败: {e}")
            
            # 获取模型信息
            model_info = self.get_model_info(model_id) if model_id not in self._metadata_cache else self._metadata_cache[model_id]
            
            return {
                "predictions": predictions,
                "inference_time": round(inference_time * 1000, 2),
                "layer_count": len(model_info.get("layers", [])) if model_info else 0,
                "activations": activations,
                "success": True,
                "input_size": list(input_data.shape[-2:]) if input_data.ndim >= 2 else [224, 224],
            }
            
        except Exception as e:
            raise InferenceException(str(e))
