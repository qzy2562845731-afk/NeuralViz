import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import time
import json
from pathlib import Path
from .base import BaseModelAdapter

class PyTorchAdapter(BaseModelAdapter):
    """
    PyTorch 模型适配器
    支持 .pt, .pth 格式及 Keras .h5 转换
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[torch.nn.Module] = None
        self.example_input: Optional[torch.Tensor] = None
        self.hooks: List = []
        self._intermediate_features: Dict[str, torch.Tensor] = {}
        self.device = torch.device("cpu")
    
    def load_model(self, file_path: str) -> bool:
        """加载 PyTorch 模型"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in [".pt", ".pth"]:
                # 加载完整模型或 state_dict
                checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        self.model = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        # 需要构建模型结构
                        raise Exception("需要模型结构定义才能加载 state_dict")
                    else:
                        # 尝试作为 state_dict 加载
                        self.model = checkpoint
                else:
                    self.model = checkpoint
                
            elif file_ext in [".h5", ".keras", ".hdf5"]:
                # Keras 模型需要先转换为 ONNX
                raise Exception("Keras 模型请先转换为 ONNX 格式")
            
            elif file_ext in [".pickle", ".pkl"]:
                checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
                self.model = checkpoint
            
            # 设置为评估模式
            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
            
            self.model_path = file_path
            self.model_name = Path(file_path).stem
            
            # 解析层信息
            self._parse_layers_from_torch()
            
            return True
        except Exception as e:
            raise Exception(f"加载 PyTorch 模型失败: {str(e)}")
    
    def _parse_layers_from_torch(self):
        """从 PyTorch 模型解析层信息（包含形状推断）"""
        self._layer_info = []
        
        if not isinstance(self.model, torch.nn.Module):
            return
        
        # 收集所有叶子模块（不包含容器）
        leaf_modules = []
        for name, module in self.model.named_modules():
            if name == "":
                continue
            # 只添加叶子模块（没有子模块的层）
            if len(list(module.children())) == 0:
                leaf_modules.append((name, module))
        
        # 用于存储每层的输入输出形状
        shapes: Dict[str, Dict[str, Any]] = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                shape_info = {}
                if input and len(input) > 0 and isinstance(input[0], torch.Tensor):
                    shape_info["input_shape"] = list(input[0].shape)
                if isinstance(output, torch.Tensor):
                    shape_info["output_shape"] = list(output.shape)
                elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    shape_info["output_shape"] = list(output[0].shape)
                shapes[name] = shape_info
            return hook
        
        # 注册钩子
        for name, module in leaf_modules:
            hook_handle = module.register_forward_hook(make_hook(name))
            hooks.append(hook_handle)
        
        # 执行一次前向传播以获取形状
        try:
            input_shape = self.get_input_shape()
            x = torch.randn(*input_shape)
            with torch.no_grad():
                self.model(x)
        except Exception:
            pass  # 如果前向传播失败，继续使用默认形状
        finally:
            # 移除钩子
            for hook in hooks:
                hook.remove()
        
        # 构建层信息
        layer_id = 0
        
        # 添加输入层
        first_module = leaf_modules[0][1] if leaf_modules else None
        input_shape_list = list(self.get_input_shape())
        # 将 NCHW 转换为 HWC 格式以便前端显示
        if len(input_shape_list) == 4:
            # [N, C, H, W] -> [H, W, C]
            input_hwc = [input_shape_list[2], input_shape_list[3], input_shape_list[1]]
        elif len(input_shape_list) == 3:
            input_hwc = input_shape_list
        elif len(input_shape_list) == 2:
            input_hwc = [input_shape_list[1], 1, 1]
        else:
            input_hwc = [28, 28, 1]
        
        self._layer_info.append({
            "id": layer_id,
            "name": "Input",
            "type": "Input",
            "params": 0,
            "input_shape": input_hwc,
            "output_shape": input_hwc,
        })
        layer_id += 1
        
        # 添加各层
        for name, module in leaf_modules:
            shape_info = shapes.get(name, {})
            input_shape = shape_info.get("input_shape")
            output_shape = shape_info.get("output_shape")
            
            # 转换为 HWC 格式
            def to_hwc(shape):
                if shape is None:
                    return None
                if len(shape) == 4:
                    return [shape[2], shape[3], shape[1]]
                elif len(shape) == 3:
                    return list(shape)
                elif len(shape) == 2:
                    return [shape[1], 1, 1]
                return list(shape)
            
            layer_info = {
                "id": layer_id,
                "name": name,
                "type": module.__class__.__name__,
                "params": 0,
                "input_shape": to_hwc(input_shape),
                "output_shape": to_hwc(output_shape),
            }
            
            # 计算参数量
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_info["params"] = params
            
            # 获取层类型特定信息
            if isinstance(module, torch.nn.Conv2d):
                layer_info["filters"] = module.out_channels
                layer_info["kernel_size"] = list(module.kernel_size)
                layer_info["stride"] = list(module.stride)
                layer_info["padding"] = list(module.padding)
                if module.bias is not None:
                    layer_info["activation"] = None
            elif isinstance(module, torch.nn.Linear):
                layer_info["out_features"] = module.out_features
                layer_info["in_features"] = module.in_features
            elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU,
                                       torch.nn.ELU, torch.nn.SELU, torch.nn.GELU,
                                       torch.nn.SiLU, torch.nn.Mish, torch.nn.Hardswish)):
                layer_info["activation"] = module.__class__.__name__
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.LayerNorm,
                                       torch.nn.GroupNorm, torch.nn.InstanceNorm2d)):
                pass  # 归一化层
            elif isinstance(module, torch.nn.Dropout):
                layer_info["dropout_p"] = module.p
            elif isinstance(module, (torch.nn.MaxPool2d, torch.nn.AvgPool2d,
                                       torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveAvgPool2d)):
                if hasattr(module, 'kernel_size'):
                    ks = module.kernel_size
                    if isinstance(ks, int):
                        layer_info["kernel_size"] = [ks, ks]
                    else:
                        layer_info["kernel_size"] = list(ks)
            
            self._layer_info.append(layer_info)
            layer_id += 1
        
        # 添加输出层（如果最后一层不是分类层）
        if self._layer_info:
            last_layer = self._layer_info[-1]
            last_type = last_layer["type"].lower()
            if last_type not in ("softmax", "logsoftmax"):
                output_shape_list = list(self.get_output_shape())
                if len(output_shape_list) == 2:
                    output_hwc = [output_shape_list[1], 1, 1]
                elif len(output_shape_list) == 3:
                    output_hwc = list(output_shape_list)
                else:
                    output_hwc = [10, 1, 1]
                
                self._layer_info.append({
                    "id": layer_id,
                    "name": "Output",
                    "type": "Output",
                    "params": 0,
                    "input_shape": output_hwc,
                    "output_shape": output_hwc,
                })
    
    def _make_hook(self, name: str):
        """创建前向钩子函数"""
        def hook(module, input, output):
            self._intermediate_features[name] = output.detach()
        return hook
    
    def _register_hooks(self, layer_name: str):
        """注册前向钩子以获取激活值"""
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(self.model, torch.nn.Module):
                hook_handle = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook_handle)
                break
    
    def get_layer_info(self) -> List[Dict[str, Any]]:
        return self._layer_info
    
    def _infer_input_shape(self) -> Tuple[int, ...]:
        """推断合理的输入形状用于前向传播"""
        if not isinstance(self.model, torch.nn.Module):
            return (1, 3, 224, 224)
        
        # 找到第一个 Conv2d 或 Linear 层
        first_conv = None
        first_linear = None
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) and first_conv is None:
                first_conv = module
            elif isinstance(module, torch.nn.Linear) and first_linear is None:
                first_linear = module
        
        if first_conv is not None:
            # CNN 模型：使用 224x224 作为默认输入尺寸
            # 如果通道数较小（如 1-4），可能是简单分类任务，用 28x28
            if first_conv.in_channels <= 4:
                return (1, first_conv.in_channels, 28, 28)
            else:
                return (1, first_conv.in_channels, 224, 224)
        elif first_linear is not None:
            # 纯全连接模型
            return (1, first_linear.in_features)
        
        return (1, 3, 224, 224)
    
    def get_input_shape(self) -> Tuple[int, ...]:
        if self.example_input is not None:
            return tuple(self.example_input.shape)
        
        return self._infer_input_shape()
    
    def get_output_shape(self) -> Tuple[int, ...]:
        if isinstance(self.model, torch.nn.Module):
            # 尝试推断输出形状
            x = torch.randn(1, *self.get_input_shape()[1:])
            with torch.no_grad():
                try:
                    output = self.model(x)
                    return tuple(output.shape)
                except:
                    pass
        return (1, 1000)  # 默认 ImageNet 1000 类
    
    def infer(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行推理"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 预处理输入
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        # 转换为 PyTorch 张量
        x = torch.from_numpy(input_data).float()
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(x)
        inference_time = time.time() - start_time
        
        # 转换回 numpy
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.numpy()
        
        return outputs, inference_time
    
    def get_activations(self, layer_name: str, input_data: np.ndarray) -> np.ndarray:
        """获取指定层激活值"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 清除旧的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._intermediate_features = {}
        
        # 注册新钩子
        self._register_hooks(layer_name)
        
        # 执行前向传播
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        x = torch.from_numpy(input_data).float()
        
        with torch.no_grad():
            self.model(x)
        
        # 获取激活值
        if layer_name in self._intermediate_features:
            activation = self._intermediate_features[layer_name]
            if isinstance(activation, torch.Tensor):
                return activation.numpy()
            return activation
        
        raise ValueError(f"未找到层: {layer_name}")
    
    def get_all_activations(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有已注册层的激活值"""
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 清除旧的钩子
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._intermediate_features = {}
        
        # 为所有命名模块注册钩子
        for name, module in self.model.named_modules():
            # 只注册有实际参数的模块，跳过容器
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.MaxPool2d) or isinstance(module, torch.nn.AdaptiveAvgPool2d):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
        
        # 执行前向传播
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        x = torch.from_numpy(input_data).float()
        
        with torch.no_grad():
            self.model(x)
        
        # 转换为 numpy
        result = {}
        for name, act in self._intermediate_features.items():
            if isinstance(act, torch.Tensor):
                result[name] = act.numpy()
            else:
                result[name] = act
        
        return result