import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import time
from pathlib import Path

import onnxruntime as ort

from .base import BaseModelAdapter

# onnx 为可选依赖，用于模型结构解析；不可用时降级为 onnxruntime 基础模式
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ONNXAdapter(BaseModelAdapter):
    """
    ONNX 模型适配器
    基于 onnxruntime 实现模型加载与推理
    onnx 包可用时提供完整层解析，否则降级为基础模式
    """

    def __init__(self):
        super().__init__()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self._input_shape: Tuple[int, ...] = ()
        self._output_shape: Tuple[int, ...] = ()
        self.intermediate_outputs: Dict[str, np.ndarray] = {}
        self._onnx_model = None

    def load_model(self, file_path: str) -> bool:
        """加载 ONNX 模型

        加载时自动检测：
        1. 输入布局（NCHW/NHWC）—— 写入 self.input_layout
        2. 输出是否已是概率（计算图末尾是否含 Softmax 算子）—— 写入 self.output_is_probability
        这两个属性供推理 API 决定预处理布局与 softmax 后处理，替代脆弱的启发式。
        """
        try:
            # 优先使用 onnx 进行模型解析与校验
            if ONNX_AVAILABLE:
                self._onnx_model = onnx.load(file_path)
                onnx.checker.check_model(self._onnx_model)
                self.model_name = self._onnx_model.graph.name or Path(file_path).stem
            else:
                self._onnx_model = None
                self.model_name = Path(file_path).stem

            # 创建推理会话（onnxruntime 始终可用）
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(
                file_path,
                providers=providers,
            )

            # 获取输入输出信息
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()

            if len(inputs) > 0:
                self.input_name = inputs[0].name
                self._input_shape = self._resolve_shape(inputs[0].shape)

            if len(outputs) > 0:
                self.output_name = outputs[0].name
                self._output_shape = self._resolve_shape(outputs[0].shape)

            self.model_path = file_path

            # 自动检测输入布局（NCHW vs NHWC）
            self.input_layout = self._detect_input_layout()

            # 自动检测输出是否已是概率（计算图末尾是否含 Softmax/LogSoftmax 节点）
            self.output_is_probability = self._detect_output_has_softmax()

            # 解析层信息
            self._parse_layers()

            return True
        except Exception as e:
            raise RuntimeError(f"加载 ONNX 模型失败: {str(e)}") from e

    def _detect_input_layout(self) -> str:
        """检测 ONNX 模型输入布局是 NCHW 还是 NHWC

        启发式：分析 input_shape 的维度特征
        - 4D shape [N, C, H, W]: shape[1] 为小整数(1/3/4)，shape[2]/[3] 较大 → NCHW
        - 4D shape [N, H, W, C]: shape[-1] 为小整数(1/3/4)，shape[1]/[2] 较大 → NHWC
        - 无法判断时默认 NCHW（PyTorch 导出的 ONNX 标准布局）
        """
        shape = list(self._input_shape)
        SMALL_CHANNELS = {1, 2, 3, 4}

        if len(shape) >= 4:
            c_nchw = shape[1]
            c_nhwc = shape[-1]
            is_nchw = (
                c_nchw in SMALL_CHANNELS
                and shape[2] not in SMALL_CHANNELS
                and shape[3] not in SMALL_CHANNELS
            )
            is_nhwc = (
                c_nhwc in SMALL_CHANNELS
                and shape[1] not in SMALL_CHANNELS
                and shape[2] not in SMALL_CHANNELS
            )
            if is_nhwc and not is_nchw:
                return "NHWC"
        return "NCHW"

    def _detect_output_has_softmax(self) -> bool:
        """检测 ONNX 计算图输出节点之前是否包含 Softmax 算子

        若模型末尾已含 Softmax/LogSoftmax 节点，则输出已是概率（或对数概率），
        推理 API 应跳过 softmax 后处理，避免二次 softmax 导致概率平方。
        LogSoftmax 输出虽非严格概率，但二次 softmax 会更严重地扭曲分布，
        因此检测到 LogSoftmax 时也返回 True（推理 API 不再 softmax）。
        """
        if not ONNX_AVAILABLE or self._onnx_model is None:
            return False
        try:
            graph = self._onnx_model.graph
            if len(graph.output) == 0:
                return False
            output_name = graph.output[0].name
            # 反向遍历节点，查找直接产出 output 的算子
            for node in graph.node:
                if output_name in node.output:
                    if node.op_type in ("Softmax", "LogSoftmax"):
                        return True
                    # Sigmoid 也视为已归一化（二分类）
                    if node.op_type == "Sigmoid":
                        return True
                    break
            return False
        except Exception:
            return False

    @staticmethod
    def _resolve_shape(raw_shape) -> Tuple[int, ...]:
        """解析 onnxruntime 返回的 shape，处理动态维度"""
        resolved = []
        for dim in raw_shape:
            if isinstance(dim, str) or dim is None:
                resolved.append(-1)
            else:
                resolved.append(int(dim))
        return tuple(resolved)

    def _parse_layers(self):
        """解析层信息，优先使用 onnx，否则使用 onnxruntime 基础信息"""
        self._layer_info = []

        if ONNX_AVAILABLE and self._onnx_model is not None:
            self._parse_layers_from_onnx(self._onnx_model)
        else:
            self._parse_layers_from_runtime()

    def _parse_layers_from_onnx(self, onnx_model):
        """从 ONNX 模型完整解析层信息"""
        try:
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        except Exception:
            pass
        
        def get_shape_from_value_info(name):
            for vi in onnx_model.graph.value_info:
                if vi.name == name:
                    try:
                        shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                        if all(s > 0 for s in shape):
                            return shape
                    except Exception:
                        pass
            return None
        
        def get_input_shape(name):
            for inp in onnx_model.graph.input:
                if inp.name == name:
                    try:
                        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                        if all(s > 0 for s in shape):
                            return shape
                    except Exception:
                        pass
            return None
        
        def get_output_shape(name):
            for out in onnx_model.graph.output:
                if out.name == name:
                    try:
                        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
                        if all(s > 0 for s in shape):
                            return shape
                    except Exception:
                        pass
            return None
        
        def to_hwc(shape):
            if shape is None:
                return None
            if len(shape) == 4:
                return [shape[2], shape[3], shape[1]]
            elif len(shape) == 3:
                return [shape[1], shape[2], shape[0]]
            elif len(shape) == 2:
                return [shape[1], 1, 1]
            return shape
        
        layer_id = 0
        
        # 添加输入层
        if len(onnx_model.graph.input) > 0:
            inp = onnx_model.graph.input[0]
            input_shape_nchw = get_input_shape(inp.name)
            input_hwc = to_hwc(input_shape_nchw) if input_shape_nchw else [28, 28, 3]
            self._layer_info.append({
                "id": layer_id,
                "name": "Input",
                "type": "Input",
                "params": 0,
                "input_shape": input_hwc,
                "output_shape": input_hwc,
            })
            layer_id += 1
        
        # 解析各节点
        for node in onnx_model.graph.node:
            layer_info: Dict[str, Any] = {
                "id": layer_id,
                "name": node.name or f"layer_{layer_id}",
                "type": node.op_type,
                "params": 0,
                "input_shape": None,
                "output_shape": None,
            }
            
            input_shape = None
            if len(node.input) > 0:
                input_shape = get_shape_from_value_info(node.input[0])
                if input_shape is None:
                    input_shape = get_input_shape(node.input[0])
            
            output_shape = None
            if len(node.output) > 0:
                output_shape = get_shape_from_value_info(node.output[0])
                if output_shape is None:
                    output_shape = get_output_shape(node.output[0])
            
            layer_info["input_shape"] = to_hwc(input_shape)
            layer_info["output_shape"] = to_hwc(output_shape)
            
            if node.op_type == "Conv":
                layer_info["params"] = self._calc_conv_params(node, onnx_model)
                for init in onnx_model.graph.initializer:
                    if init.name == node.input[1]:
                        layer_info["filters"] = int(init.dims[0])
                        layer_info["kernel_size"] = [int(init.dims[2]), int(init.dims[3])] if len(init.dims) >= 4 else [3, 3]
                        break
            
            elif node.op_type in ("MatMul", "Gemm"):
                layer_info["params"] = self._calc_matmul_params(node, onnx_model)
                for init in onnx_model.graph.initializer:
                    if init.name == node.input[1]:
                        layer_info["out_features"] = int(init.dims[1])
                        break
            
            self._layer_info.append(layer_info)
            layer_id += 1
        
        # 添加输出层
        if len(onnx_model.graph.output) > 0:
            out = onnx_model.graph.output[0]
            output_shape_nchw = get_output_shape(out.name)
            output_hwc = to_hwc(output_shape_nchw) if output_shape_nchw else [1000, 1, 1]
            self._layer_info.append({
                "id": layer_id,
                "name": "Output",
                "type": "Output",
                "params": 0,
                "input_shape": output_hwc,
                "output_shape": output_hwc,
            })

    def _parse_layers_from_runtime(self):
        """基于 onnxruntime 构建基础层信息（降级方案）"""
        # 输入层
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        layer_id = 0

        for inp in inputs:
            self._layer_info.append({
                "id": layer_id,
                "name": inp.name,
                "type": "Input",
                "params": 0,
                "input_shape": None,
                "output_shape": list(self._resolve_shape(inp.shape)),
            })
            layer_id += 1

        # 输出层
        for out in outputs:
            self._layer_info.append({
                "id": layer_id,
                "name": out.name,
                "type": "Output",
                "params": 0,
                "input_shape": None,
                "output_shape": list(self._resolve_shape(out.shape)),
            })
            layer_id += 1

    def _calc_conv_params(self, node, onnx_model) -> int:
        """计算卷积层参数量（含偏置）"""
        params = 0
        for init in onnx_model.graph.initializer:
            if init.name == node.input[1]:
                params = int(np.prod(init.dims))
        # be15修复：包含偏置参数（bias 通常为 node.input[2]）
        if len(node.input) > 2 and node.input[2]:
            for init in onnx_model.graph.initializer:
                if init.name == node.input[2]:
                    params += int(np.prod(init.dims))
                    break
        return params

    def _calc_matmul_params(self, node, onnx_model) -> int:
        """计算矩阵乘法/全连接层参数量（含偏置）"""
        params = 0
        for init in onnx_model.graph.initializer:
            if init.name == node.input[1]:
                params = int(np.prod(init.dims))
        # be15修复：Gemm 层的 bias 通常为 node.input[2]
        if len(node.input) > 2 and node.input[2]:
            for init in onnx_model.graph.initializer:
                if init.name == node.input[2]:
                    params += int(np.prod(init.dims))
                    break
        return params

    def get_layer_info(self) -> List[Dict[str, Any]]:
        return self._layer_info

    def get_input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def get_output_shape(self) -> Tuple[int, ...]:
        return self._output_shape

    def infer(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行推理

        输入数据布局应与 self.input_layout 一致（由推理 API 的 preprocess_image 保证）。
        此处仅做维度补全与类型转换，不重复 transpose，避免布局错乱。
        """
        if self.session is None:
            raise RuntimeError("模型未加载")

        # 预处理输入：补全 batch 维
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        # 转换为 float32
        input_data = input_data.astype(np.float32)

        # 推理
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_data})
        inference_time = time.time() - start_time

        return outputs[0], inference_time

    def get_activations(self, layer_name: str, input_data: np.ndarray) -> np.ndarray:
        """获取层激活值 - 通过添加中间输出节点实现"""
        if self.session is None:
            raise RuntimeError("模型未加载")

        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        input_data = input_data.astype(np.float32)

        # 重新创建会话，包含目标输出节点
        try:
            all_outputs = [self.output_name, layer_name]
            sess = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            outputs = sess.run(all_outputs, {self.input_name: input_data})
            return outputs[1]
        except Exception as e:
            raise NotImplementedError(
                f"无法获取层 {layer_name} 的激活值: {str(e)}"
            )
    
    def get_all_activations(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有层的激活值"""
        if self.session is None:
            raise RuntimeError("模型未加载")
        
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_data = input_data.astype(np.float32)
        
        activations = {}
        
        try:
            # 获取所有节点输出名称
            all_outputs = []
            if self._onnx_model is not None:
                for node in self._onnx_model.graph.node:
                    for output in node.output:
                        all_outputs.append(output)
            
            # 分批获取激活值，避免会话过大
            batch_size = 20
            for i in range(0, len(all_outputs), batch_size):
                batch = all_outputs[i:i + batch_size]
                try:
                    sess = ort.InferenceSession(
                        self.model_path,
                        providers=["CPUExecutionProvider"],
                    )
                    outputs = sess.run(batch, {self.input_name: input_data})
                    for name, out in zip(batch, outputs):
                        activations[name] = out
                except Exception:
                    pass
        except Exception as e:
            print(f"ONNX 激活值提取失败: {e}")
        
        return activations
