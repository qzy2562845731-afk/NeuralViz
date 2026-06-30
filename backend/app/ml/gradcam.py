"""
Grad-CAM 显著性可视化模块
实现 Grad-CAM、Grad-CAM++ 算法，用于模型决策可视化解释
支持 PyTorch CNN 模型的热力图生成
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import io
import base64


class GradCAM:
    """Grad-CAM 实现

    论文: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ 实现（改进版本，更好的定位能力）

    论文: Grad-CAM++: Improved Visual Explanations from Deep Networks
    https://arxiv.org/abs/1710.11063
    """

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        grads = self.gradients
        acts = self.activations

        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3

        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7
        alpha = alpha_num / alpha_denom

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def find_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """自动查找模型中最后一个卷积层作为目标层"""
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if conv_layers:
        return conv_layers[-1][1]
    return None


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """将灰度热力图 [0,1] 映射为 JET 彩色 RGB 图像 (H,W,3) [0,255]"""
    h, w = heatmap.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    hm = np.clip(heatmap, 0.0, 1.0)

    result[:, :, 0] = np.clip(1.5 - np.abs(4 * hm - 3), 0, 1) * 255
    result[:, :, 1] = np.clip(1.5 - np.abs(4 * hm - 2), 0, 1) * 255
    result[:, :, 2] = np.clip(1.5 - np.abs(4 * hm - 1), 0, 1) * 255

    return result


def _resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """使用numpy进行简单的图像缩放（最近邻）"""
    from PIL import Image
    if img.ndim == 2:
        pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8), mode='L')
    elif img.shape[2] == 1:
        pil_img = Image.fromarray((img[:, :, 0] * 255).astype(np.uint8) if img.max() <= 1.0 else img[:, :, 0].astype(np.uint8), mode='L')
    else:
        pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8), mode='RGB')
    pil_img = pil_img.resize(size, Image.BILINEAR)
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """将热力图叠加到原始图像上

    Args:
        image: 原始图像 (H, W, C) 或 (H, W)，uint8 或 float [0,1]
        heatmap: 热力图 (H, W)，float [0,1]
        alpha: 热力图透明度

    Returns:
        叠加后的图像 (H, W, 3)，uint8 [0, 255]
    """
    from PIL import Image

    if image.max() <= 1.0 and image.dtype in (np.float32, np.float64):
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    if image_uint8.ndim == 2:
        image_uint8 = np.stack([image_uint8] * 3, axis=-1)
    elif image_uint8.shape[2] == 1:
        image_uint8 = np.concatenate([image_uint8] * 3, axis=-1)
    elif image_uint8.shape[2] == 4:
        image_uint8 = image_uint8[:, :, :3]

    h, w = image_uint8.shape[:2]

    heatmap_resized = _resize_image(heatmap[:, :, np.newaxis], (w, h))[:, :, 0]
    heatmap_colored = _apply_colormap(heatmap_resized.astype(np.float32) / 255.0)

    overlaid = (image_uint8.astype(np.float32) * (1 - alpha) + heatmap_colored.astype(np.float32) * alpha).astype(np.uint8)

    return overlaid


def _encode_png_base64(img_rgb: np.ndarray) -> str:
    """将RGB numpy数组编码为PNG base64"""
    from PIL import Image
    pil_img = Image.fromarray(img_rgb, mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_gradcam_visualization(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    use_plusplus: bool = True,
    alpha: float = 0.4,
) -> Dict[str, Any]:
    """生成 Grad-CAM 可视化结果

    Args:
        model: PyTorch CNN 模型
        input_tensor: 预处理后的输入张量 (1, C, H, W)
        original_image: 原始图像（用于叠加）
        target_class: 目标类别
        use_plusplus: 是否使用 Grad-CAM++
        alpha: 热力图透明度

    Returns:
        dict with keys: heatmap_base64, overlaid_base64, predicted_class, etc.
    """
    target_layer = find_target_layer(model)
    if target_layer is None:
        return {"error": "模型中未找到卷积层，无法生成 Grad-CAM"}

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()

    if target_class is None:
        target_class = pred_class

    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    cam_extractor = CAMClass(model, target_layer)

    try:
        heatmap = cam_extractor.generate(input_tensor, target_class)
    finally:
        cam_extractor.remove_hooks()

    if original_image.ndim == 3 and original_image.shape[0] in (1, 3, 4) and original_image.shape[-1] not in (1, 3, 4):
        original_image = np.transpose(original_image, (1, 2, 0))

    if original_image.max() <= 1.0:
        orig_disp = (original_image * 255).astype(np.uint8)
    else:
        orig_disp = original_image.astype(np.uint8)

    if orig_disp.ndim == 2:
        orig_disp = np.stack([orig_disp] * 3, axis=-1)
    elif orig_disp.shape[2] == 1:
        orig_disp = np.concatenate([orig_disp] * 3, axis=-1)
    elif orig_disp.shape[2] == 4:
        orig_disp = orig_disp[:, :, :3]

    overlaid = overlay_heatmap(orig_disp, heatmap, alpha=alpha)
    heatmap_colored = _apply_colormap(heatmap)

    if heatmap_colored.shape[:2] != orig_disp.shape[:2]:
        heatmap_colored = _resize_image(heatmap_colored, orig_disp.shape[:2][::-1])

    overlaid_b64 = _encode_png_base64(overlaid)
    heatmap_b64 = _encode_png_base64(heatmap_colored)
    original_b64 = _encode_png_base64(orig_disp)

    return {
        "original_image": f"data:image/png;base64,{original_b64}",
        "heatmap": f"data:image/png;base64,{heatmap_b64}",
        "overlay": f"data:image/png;base64,{overlaid_b64}",
        "heatmap_base64": heatmap_b64,
        "overlaid_base64": overlaid_b64,
        "predicted_class": int(pred_class),
        "confidence": round(float(pred_prob), 4),
        "predicted_probability": round(float(pred_prob), 4),
        "target_class": int(target_class),
        "method": "Grad-CAM++" if use_plusplus else "Grad-CAM",
    }
