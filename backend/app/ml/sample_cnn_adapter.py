"""
内置示例 CNN 模型适配器
基于 PyTorch SimpleCNN 实现真实推理，自动加载已训练好的最佳模型权重
用于开箱即用的演示和测试
"""
import time
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseModelAdapter
from .model_builder import SimpleCNN


class SampleCNNAdapter(BaseModelAdapter):
    """
    内置示例 CNN 适配器
    基于真实 PyTorch SimpleCNN 模型，自动加载训练好的权重；
    无训练权重时回退到随机权重，但 PyTorch 卷积运算仍远快于 numpy 实现。
    """

    def __init__(self):
        super().__init__()
        self.model_name = "SampleCNN (MNIST)"
        self._layer_info: List[Dict[str, Any]] = []
        self._input_shape: Tuple[int, ...] = (1, 1, 28, 28)
        self._output_shape: Tuple[int, ...] = (1, 10)
        self._activations: Dict[str, np.ndarray] = {}
        # PyTorch 模型及钩子相关
        self._torch_model: Optional[torch.nn.Module] = None
        self._hooks: List = []
        self._intermediate_features: Dict[str, torch.Tensor] = {}
        self.device = torch.device("cpu")
        # SampleCNN 内部 infer 已应用 torch.softmax，输出是概率分布
        # 推理 API 据此跳过 softmax 后处理，避免二次 softmax
        self.output_is_probability: bool = True
        # SimpleCNN 是 PyTorch 模型，输入布局固定为 NCHW
        self.input_layout: str = "NCHW"

    def load_model(self, file_path: str = "") -> bool:
        """加载示例模型：创建 SimpleCNN 并尝试加载训练好的权重"""
        self.model_path = file_path or "sample_cnn"
        self.model_name = "SampleCNN (MNIST)"

        # 1. 创建 SimpleCNN 模型（MNIST：1 通道，10 类，28x28 输入）
        self._torch_model = SimpleCNN(in_channels=1, num_classes=10, input_size=28)

        # 2. 尝试从数据库加载训练好的权重
        self._try_load_trained_weights()

        # 3. 设置为评估模式（关闭 Dropout）
        self._torch_model.eval()

        # 4. 构建层信息（前端兼容格式）
        self._build_layer_info()
        return True

    def _try_load_trained_weights(self) -> bool:
        """从数据库查找最佳已完成实验，加载其模型权重

        遍历已完成实验（按准确率降序），找到第一个拥有可加载 best.pt 的实验。
        某些实验可能缺少权重文件或结构不匹配，逐个尝试以保证优雅降级。
        """
        try:
            from app.core.database import SessionLocal
            from app.models.experiment import Experiment
            from app.core.config import settings

            db = SessionLocal()
            try:
                # 查询所有已完成实验（按准确率降序）
                exps = db.query(Experiment).filter(
                    Experiment.status == "completed",
                    Experiment.is_deleted == False,
                ).order_by(
                    Experiment.best_accuracy.desc()
                ).limit(20).all()

                for exp in exps:
                    best_pt = settings.MODEL_DIR / exp.experiment_id / "best.pt"
                    if not best_pt.exists():
                        continue
                    try:
                        state_dict = torch.load(
                            str(best_pt),
                            map_location="cpu",
                            weights_only=True,
                        )
                        self._torch_model.load_state_dict(state_dict)
                        self.model_name = f"Trained CNN (acc={exp.best_accuracy:.2%})"
                        return True
                    except Exception as load_err:
                        # 权重结构与当前模型不匹配，尝试下一个实验
                        print(f"[SampleCNNAdapter] 跳过 {exp.experiment_id}: {load_err}")
                        continue
            finally:
                db.close()
            return False
        except Exception as e:
            print(f"[SampleCNNAdapter] 加载训练权重失败: {e}")
            return False

    def _build_layer_info(self):
        """构建 CNN 层信息（与前端 modelAnalyzer 格式完全兼容）

        前端支持的 layer type: input, conv, pool, fc, output, norm, dropout
        字段名使用 camelCase 以匹配前端 parseLayerItem 的解析逻辑
        层参数与 SimpleCNN 实际结构一致（padding=1 保持尺寸，AdaptiveAvgPool 到 4x4）
        """
        self._layer_info = [
            {
                "id": "input",
                "name": "Input",
                "type": "input",
                "params": 0,
                "nodeCount": 1,
                "inputShape": [28, 28, 1],
                "outputShape": [28, 28, 1],
            },
            {
                "id": "conv1",
                "name": "Conv1",
                "type": "conv",
                "params": 320,
                "nodeCount": 32,
                "inputShape": [28, 28, 1],
                "outputShape": [28, 28, 32],
                "kernelSize": 3,
                "activation": "relu",
            },
            {
                "id": "pool1",
                "name": "MaxPool1",
                "type": "pool",
                "params": 0,
                "nodeCount": 32,
                "inputShape": [28, 28, 32],
                "outputShape": [14, 14, 32],
                "kernelSize": 2,
            },
            {
                "id": "conv2",
                "name": "Conv2",
                "type": "conv",
                "params": 18496,
                "nodeCount": 64,
                "inputShape": [14, 14, 32],
                "outputShape": [14, 14, 64],
                "kernelSize": 3,
                "activation": "relu",
            },
            {
                "id": "pool2",
                "name": "MaxPool2",
                "type": "pool",
                "params": 0,
                "nodeCount": 64,
                "inputShape": [14, 14, 64],
                "outputShape": [7, 7, 64],
                "kernelSize": 2,
            },
            {
                "id": "fc1",
                "name": "FC1",
                "type": "fc",
                "params": 131200,
                "nodeCount": 128,
                "inputShape": [1024],
                "outputShape": [128],
                "activation": "relu",
            },
            {
                "id": "output",
                "name": "Output",
                "type": "output",
                "params": 1290,
                "nodeCount": 10,
                "inputShape": [128],
                "outputShape": [10],
                "activation": "softmax",
            },
        ]

    def _make_hook(self, name: str):
        """创建前向钩子函数，将层输出存入 _intermediate_features"""
        def hook(module, inp, output):
            self._intermediate_features[name] = output.detach()
        return hook

    def _register_hooks(self):
        """注册前向钩子以提取各层激活值"""
        # 清除旧钩子
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._intermediate_features = {}

        # SimpleCNN 模块名 -> 友好层名 的映射
        name_mapping = {
            "features.0": "conv1",       # Conv2d
            "features.1": "relu1",       # ReLU
            "features.2": "pool1",       # MaxPool2d
            "features.3": "conv2",       # Conv2d
            "features.4": "relu2",       # ReLU
            "features.5": "pool2",       # MaxPool2d
            "features.6": "adaptive_pool",  # AdaptiveAvgPool2d
            "classifier.0": "flatten",   # Flatten
            "classifier.1": "fc1",       # Linear
            "classifier.2": "relu3",     # ReLU
            "classifier.4": "output",    # Linear (输出层)
        }

        for name, module in self._torch_model.named_modules():
            friendly = name_mapping.get(name)
            if friendly:
                hook = module.register_forward_hook(self._make_hook(friendly))
                self._hooks.append(hook)

    def get_layer_info(self) -> List[Dict[str, Any]]:
        return self._layer_info

    def get_input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def get_output_shape(self) -> Tuple[int, ...]:
        return self._output_shape

    def infer(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """使用 PyTorch 执行真实推理，返回概率分布和推理耗时"""
        start_time = time.time()

        if self._torch_model is None:
            raise RuntimeError("模型未加载")

        # 确保是 NCHW 格式
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)
        if input_data.ndim == 4 and input_data.shape[1] > 3 and input_data.shape[-1] <= 3:
            # HWC -> CHW
            input_data = input_data.transpose(0, 3, 1, 2)

        # 调整到 1 通道 28x28
        x = self._preprocess_input(input_data)

        # 注册钩子以提取激活值
        self._register_hooks()

        # PyTorch 推理
        x_tensor = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self._torch_model(x_tensor)

        # Softmax 得到概率分布
        probs = torch.softmax(logits, dim=1)
        output = probs.numpy()

        # 收集激活值
        self._activations = {"input": x.copy()}
        for name, act in self._intermediate_features.items():
            if isinstance(act, torch.Tensor):
                self._activations[name] = act.cpu().numpy()
        self._activations["output"] = output.copy()

        # 移除钩子
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        inference_time = time.time() - start_time
        return output, inference_time

    def _preprocess_input(self, x: np.ndarray) -> np.ndarray:
        """将输入调整为 1x1x28x28 的 MNIST 格式

        从 inference API 调用时，输入已经过 preprocess_image 处理（仅 /255 归一化到 [0,1]），
        此处只做格式保证（灰度、尺寸），不重复归一化以免破坏输入分布。

        注意：resize 分支使用 PyTorch 的 interpolate 而非 PIL.Image.fromarray(uint8)，
        避免将 [0,1] float 数据先反归一化到 [0,255] uint8 再 /255 的量化损失与二次归一化缺陷。
        """
        # 转灰度
        if x.shape[1] == 3:
            x = np.mean(x, axis=1, keepdims=True)
        elif x.shape[1] > 3:
            x = x[:, :1, :, :]

        # Resize 到 28x28（仅在尺寸不匹配时）
        N, C, H, W = x.shape
        if H != 28 or W != 28:
            # 使用 PyTorch interpolate 在 float 域直接 resize，
            # 避免 PIL uint8 量化与重归一化导致的输入分布破坏
            import torch.nn.functional as F
            x_tensor = torch.from_numpy(x).float()
            # align_corners=False 与 PIL BILINEAR 行为接近
            x_tensor = F.interpolate(
                x_tensor,
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            )
            x = x_tensor.numpy()

        return x.astype(np.float32)

    def get_activations(self, layer_name: str, input_data: np.ndarray) -> np.ndarray:
        """获取指定层的激活值"""
        # 如果还没推理过，先推理
        if not self._activations:
            self.infer(input_data)

        if layer_name in self._activations:
            return self._activations[layer_name]

        # 尝试模糊匹配
        for key in self._activations:
            if layer_name.lower() in key.lower() or key.lower() in layer_name.lower():
                return self._activations[key]

        raise ValueError(f"未找到层: {layer_name}")

    def get_all_activations(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有层的激活值"""
        if not self._activations:
            self.infer(input_data)
        return self._activations.copy()
