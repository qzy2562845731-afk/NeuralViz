"""
Grad-CAM 可视化 API 路由
前缀 /api/gradcam
"""
import io
import base64
import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from typing import Optional

from app.core.security import sanitize_path_id
from app.ml.gradcam import (
    generate_gradcam_visualization,
    find_target_layer,
    GradCAM,
    GradCAMPlusPlus,
    overlay_heatmap,
    _apply_colormap,
)

router = APIRouter()


@router.post("/generate/{experiment_id}")
async def generate_gradcam(
    experiment_id: str,
    file: UploadFile = File(..., description="输入图片"),
    target_class: Optional[int] = Form(None, description="目标类别（None则使用预测类别）"),
    use_plusplus: bool = Form(True, description="是否使用Grad-CAM++"),
    alpha: float = Form(0.4, ge=0.0, le=1.0, description="热力图透明度"),
    layer_index: int = Form(-1, description="目标卷积层索引（-1表示最后一层）"),
):
    """为指定实验的训练模型生成 Grad-CAM 可视化

    - 加载训练好的模型
    - 对输入图片进行前向传播
    - 生成热力图并叠加到原始图片
    - 返回 base64 编码的图片
    """
    import torch
    from app.core.config import settings
    from app.services.experiment_service import ExperimentService
    from app.ml.model_builder import build_model

    experiment_id = sanitize_path_id(experiment_id)

    # 加载实验信息
    exp_service = ExperimentService()
    exp_info = exp_service.get_experiment(experiment_id)
    if not exp_info:
        raise HTTPException(status_code=404, detail=f"实验 {experiment_id} 不存在")

    # 加载模型
    model_path = settings.MODEL_DIR / experiment_id / "best_model.pt"
    if not model_path.exists():
        # 尝试 final_model.pt
        model_path = settings.MODEL_DIR / experiment_id / "final_model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="训练模型文件不存在，请确保训练已完成并保存了模型")

    config = exp_info.get("config", {})
    dataset_type = config.get("dataset_type", "image_folder")
    feature_shape = config.get("feature_shape", "1x28x28")
    num_classes = exp_info.get("num_classes", 10)
    model_config = config.get("model_config", {})

    try:
        model = build_model(dataset_type, feature_shape, num_classes, model_config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

    # 读取图片
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_array = np.array(original_image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图片读取失败: {str(e)}")
    finally:
        await file.close()

    # 预处理图片
    from app.ml.model_builder import _parse_image_shape
    c, h, w = _parse_image_shape(feature_shape)
    img_resized = original_image.resize((w, h), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    # 处理通道：确保是 (C, H, W)
    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, :, :]
    elif img_array.shape[-1] in (1, 3, 4):
        img_array = np.transpose(img_array, (2, 0, 1))

    # 灰度图转3通道或单通道
    if c == 1 and img_array.shape[0] == 3:
        img_array = np.mean(img_array, axis=0, keepdims=True)
    elif c == 3 and img_array.shape[0] == 1:
        img_array = np.repeat(img_array, 3, axis=0)

    input_tensor = torch.from_numpy(img_array).unsqueeze(0).float()

    # 生成 Grad-CAM
    try:
        result = generate_gradcam_visualization(
            model=model,
            input_tensor=input_tensor,
            original_image=original_array,
            target_class=target_class,
            use_plusplus=use_plusplus,
            alpha=alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM生成失败: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "code": 200,
        "message": "Grad-CAM可视化生成成功",
        "data": result,
    }


@router.post("/compare/{experiment_id}")
async def compare_gradcam_layers(
    experiment_id: str,
    file: UploadFile = File(..., description="输入图片"),
    use_plusplus: bool = Form(True),
    alpha: float = Form(0.4),
):
    """生成所有卷积层的 Grad-CAM 对比

    返回每层卷积的 Grad-CAM 热力图，便于对比不同层的关注区域
    """
    import torch
    from app.core.config import settings
    from app.services.experiment_service import ExperimentService
    from app.ml.model_builder import build_model

    experiment_id = sanitize_path_id(experiment_id)

    exp_service = ExperimentService()
    exp_info = exp_service.get_experiment(experiment_id)
    if not exp_info:
        raise HTTPException(status_code=404, detail=f"实验 {experiment_id} 不存在")

    model_path = settings.MODEL_DIR / experiment_id / "best_model.pt"
    if not model_path.exists():
        model_path = settings.MODEL_DIR / experiment_id / "final_model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="训练模型文件不存在")

    config = exp_info.get("config", {})
    dataset_type = config.get("dataset_type", "image_folder")
    feature_shape = config.get("feature_shape", "1x28x28")
    num_classes = exp_info.get("num_classes", 10)
    model_config = config.get("model_config", {})

    try:
        model = build_model(dataset_type, feature_shape, num_classes, model_config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")

    # 读取图片
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_array = np.array(original_image)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"图片读取失败: {str(e)}")
    finally:
        await file.close()

    from app.ml.model_builder import _parse_image_shape
    c, h, w = _parse_image_shape(feature_shape)
    img_resized = original_image.resize((w, h), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = img_array[np.newaxis, :, :]
    elif img_array.shape[-1] in (1, 3, 4):
        img_array = np.transpose(img_array, (2, 0, 1))
    if c == 1 and img_array.shape[0] == 3:
        img_array = np.mean(img_array, axis=0, keepdims=True)
    elif c == 3 and img_array.shape[0] == 1:
        img_array = np.repeat(img_array, 3, axis=0)
    input_tensor = torch.from_numpy(img_array).unsqueeze(0).float()

    # 查找所有卷积层
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))

    CAMClass = GradCAMPlusPlus if use_plusplus else GradCAM
    layer_results = {}

    for layer_name, layer_module in conv_layers:
        cam_extractor = CAMClass(model, layer_module)
        try:
            heatmap = cam_extractor.generate(input_tensor)
        except Exception:
            heatmap = np.zeros((h, w), dtype=np.float32)
        finally:
            cam_extractor.remove_hooks()

        heatmap_colored = _apply_colormap(heatmap)
        pil_img = Image.fromarray(heatmap_colored, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        layer_results[layer_name] = {
            "heatmap": f"data:image/png;base64,{heatmap_b64}",
            "heatmap_base64": heatmap_b64,
        }

    # 原始图片
    buf = io.BytesIO()
    orig_pil = Image.fromarray(original_array)
    orig_pil.save(buf, format="PNG")
    buf.seek(0)
    orig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "code": 200,
        "message": "各层Grad-CAM生成成功",
        "data": {
            "original_image": f"data:image/png;base64,{orig_b64}",
            "layers": layer_results,
            "num_layers": len(layer_results),
        },
    }