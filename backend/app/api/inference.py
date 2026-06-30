import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
import io

from app.core.security import sanitize_path_id

from app.core.config import settings
from app.core.exception import ModelNotFoundException, InferenceException
from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

def preprocess_image(
    file_data: bytes,
    target_size: tuple = (224, 224),
    target_channels: int = 3,
    layout: str = "NCHW",
) -> np.ndarray:
    """图片预处理 - 与训练时保持一致的归一化策略

    关键约定：仅做 /255 归一化到 [0, 1]，不做 mean/std 标准化。
    原因：training_service._prepare_data 训练时仅 /255，若推理时额外应用
    MNIST/ImageNet 标准化会导致输入分布偏移，softmax 后概率接近均匀分布
    或错判。此函数是训练与推理共用的唯一归一化入口，确保一致性。

    Args:
        file_data: 图片字节数据
        target_size: 目标尺寸 (height, width)
        target_channels: 模型期望的通道数 (1=灰度, 3=RGB)
        layout: 输出布局 "NCHW"（PyTorch 默认）或 "NHWC"（TF/部分 ONNX）

    Returns:
        预处理后的 numpy 数组
        - NCHW: (1, C, H, W)
        - NHWC: (1, H, W, C)
    """
    # 打开图片
    image = Image.open(io.BytesIO(file_data))

    # 根据模型输入通道数自动适配
    # 单通道模型：RGB/RGBA/灰度图统一转为灰度
    # 三通道模型：灰度图自动扩展为三通道
    if target_channels == 1:
        if image.mode != "L":
            image = image.convert("L")
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")

    # 调整尺寸
    image = image.resize(target_size, Image.BILINEAR)

    # 转换为 numpy 数组
    img_array = np.array(image, dtype=np.float32)

    # 确保数组维度为 (H, W, C) 或 (H, W)
    if img_array.ndim == 2:
        if target_channels == 1:
            img_array = img_array[:, :, np.newaxis]  # (H, W, 1)
        else:
            img_array = np.stack([img_array] * 3, axis=-1)  # (H, W, 3)
    elif img_array.ndim == 3:
        actual_channels = img_array.shape[2]
        if actual_channels == 4:
            if target_channels == 1:
                img_array = np.array(image.convert("L"), dtype=np.float32)[:, :, np.newaxis]
            else:
                img_array = img_array[:, :, :3]
        elif actual_channels == 1 and target_channels == 3:
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif actual_channels == 3 and target_channels == 1:
            img_array = np.array(image.convert("L"), dtype=np.float32)[:, :, np.newaxis]

    # 归一化到 [0, 1]（与训练时一致，不做 mean/std 标准化）
    img_array = img_array / 255.0

    # 按目标布局输出
    if layout.upper() == "NHWC":
        # (H, W, C) -> (1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
    else:
        # 默认 NCHW: (H, W, C) -> (1, C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.float32)


def _detect_layout_from_shape(input_shape: list) -> tuple:
    """从模型 input_shape 自动推断布局、目标尺寸与通道数

    启发式规则：
    - 4D shape [N, C, H, W] (NCHW): shape[1] 为小整数(1/3/4)，shape[2]/[3] 较大
    - 4D shape [N, H, W, C] (NHWC): shape[-1] 为小整数(1/3/4)，shape[1]/[2] 较大
    - 3D shape [C, H, W] 或 [H, W, C] 按相同启发式处理
    - 含动态维度(-1) 时跳过该维度比较

    Returns:
        (layout, target_size, target_channels)
        layout: "NCHW" 或 "NHWC"
        target_size: (H, W)
        target_channels: int
    """
    shape = [int(s) if s is not None else -1 for s in input_shape]
    # 过滤动态维度用于通道候选判断
    fixed = [s for s in shape if s > 0]

    SMALL_CHANNELS = {1, 2, 3, 4}

    if len(shape) >= 4:
        # 4D: [N, ?, ?, ?]
        c_candidate_nchw = shape[1]
        c_candidate_nhwc = shape[-1]
        # NCHW: shape[1] 是小整数且 shape[2]/[3] 较大
        is_nchw = (
            c_candidate_nchw in SMALL_CHANNELS
            and shape[2] not in SMALL_CHANNELS
            and shape[3] not in SMALL_CHANNELS
        )
        is_nhwc = (
            c_candidate_nhwc in SMALL_CHANNELS
            and shape[1] not in SMALL_CHANNELS
            and shape[2] not in SMALL_CHANNELS
        )
        if is_nhwc and not is_nchw:
            return "NHWC", (shape[1] or 28, shape[2] or 28), c_candidate_nhwc
        # 默认 NCHW（PyTorch 标准）
        h = shape[2] if shape[2] > 0 else 28
        w = shape[3] if shape[3] > 0 else 28
        c = c_candidate_nchw if c_candidate_nchw > 0 else (c_candidate_nhwc if c_candidate_nhwc > 0 else 1)
        return "NCHW", (h, w), c
    elif len(shape) == 3:
        # 3D: [C, H, W] 或 [H, W, C]
        c_first = shape[0]
        c_last = shape[-1]
        if c_last in SMALL_CHANNELS and c_first not in SMALL_CHANNELS:
            return "NHWC", (shape[0] or 28, shape[1] or 28), c_last
        c = c_first if c_first > 0 else (c_last if c_last > 0 else 1)
        return "NCHW", (shape[1] or 28, shape[2] or 28), c
    elif len(shape) == 2:
        # 2D: [H, W] 灰度
        return "NCHW", (shape[0] or 28, shape[1] or 28), 1
    else:
        return "NCHW", (28, 28), 1


def _parse_feature_shape(feature_shape: str):
    """解析图像特征维度字符串 '28x28x1' -> ((28, 28), 1)"""
    try:
        parts = feature_shape.lower().split("x")
        if len(parts) == 3:
            w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
            return (h, w), c
        elif len(parts) == 2:
            h, w = int(parts[0]), int(parts[1])
            return (h, w), 1
    except (ValueError, IndexError):
        pass
    return (28, 28), 1


def _load_experiment_adapter(experiment_id: str):
    """从数据库加载实验对应的训练模型

    通过实验记录查找关联数据集，获取 dataset_type / feature_shape / num_classes，
    使用 load_trained_model 构建并加载训练好的权重。

    Returns:
        (adapter, model_info) 元组；加载失败返回 (None, None)
    """
    experiment_id = sanitize_path_id(experiment_id)
    import json
    from app.core.database import SessionLocal
    from app.models.experiment import Experiment
    from app.models.dataset import Dataset
    from app.ml.model_builder import load_trained_model
    from app.ml.pytorch_adapter import PyTorchAdapter

    db = SessionLocal()
    try:
        exp = db.query(Experiment).filter(
            Experiment.experiment_id == experiment_id,
            Experiment.is_deleted == False,
        ).first()
        if not exp:
            return None, None

        # 从 config 中获取 dataset_id
        config = json.loads(exp.config) if exp.config else {}
        dataset_id = config.get("dataset_id")
        if not dataset_id:
            return None, None

        # 查询数据集获取 dataset_type / feature_shape / num_classes
        ds = db.query(Dataset).filter(
            Dataset.dataset_id == dataset_id
        ).first()
        if not ds:
            return None, None

        dataset_type = ds.dataset_type or "mnist_idx"
        feature_shape = ds.feature_shape or "28x28x1"
        num_classes = ds.class_count or 10

        # 仅支持图像类数据集的图片推理
        if dataset_type not in ("image_folder", "mnist_idx"):
            return None, None

        # 构建模型权重路径
        model_path = settings.MODEL_DIR / experiment_id / "best.pt"
        if not model_path.exists():
            return None, None

        # 从 config 中提取 model_config（用于构建带注意力/通道配置的模型）
        model_config = config.get("model_config", {}) or {}

        # 加载训练好的模型
        model = load_trained_model(str(model_path), dataset_type, feature_shape, num_classes, model_config)
        model.eval()

        # 创建 PyTorchAdapter 并注入已加载的模型
        adapter = PyTorchAdapter()
        adapter.model = model
        adapter.model_path = str(model_path)
        adapter.model_name = exp.name or "Trained Model"
        adapter._parse_layers_from_torch()

        # 从 feature_shape 解析输入尺寸和通道数（比推断更准确）
        target_size, target_channels = _parse_feature_shape(feature_shape)
        model_info = {
            "model_id": experiment_id,
            "model_name": exp.name or "Trained Model",
            "input_shape": [1, target_channels, target_size[0], target_size[1]],
            "output_shape": [1, num_classes],
            "layers": adapter.get_layer_info(),
        }
        return adapter, model_info
    except Exception as e:
        print(f"[inference] 加载实验模型失败 ({experiment_id}): {e}")
        return None, None
    finally:
        db.close()


@router.post("/image")
async def inference_image(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    图片推理接口
    
    接收模型ID和图片文件，执行推理并返回预测结果和层激活值
    """
    # 特殊处理：内置示例模型
    model_lower = model_id.lower().strip()
    is_sample = model_lower in ("sample_cnn", "sample", "default", "")
    
    adapter = None
    model_info = None
    
    if is_sample:
        # 使用内置示例模型（SampleCNNAdapter 内部用 PyTorch SimpleCNN）
        try:
            from app.ml.factory import ModelAdapterFactory
            adapter = ModelAdapterFactory.create("sample_cnn")
            adapter.load_model("sample_cnn")
            model_info = {
                "model_id": "sample_cnn",
                "model_name": adapter.model_name or "SampleCNN (MNIST)",
                "input_shape": [1, 1, 28, 28],
                "layers": adapter.get_layer_info(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"示例模型加载失败: {str(e)}")
    else:
        # 优先尝试作为实验ID加载训练好的模型
        exp_adapter, exp_model_info = _load_experiment_adapter(model_id)
        if exp_adapter is not None:
            adapter = exp_adapter
            model_info = exp_model_info
        else:
            # 回退到普通模型服务
            model_info = model_service.get_model_info(model_id)
            if not model_info:
                raise HTTPException(status_code=404, detail="模型不存在或未加载")
    
    # 保存上传图片
    image_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    image_filename = f"{image_id}{file_ext}"
    image_path = settings.IMAGE_DIR / image_filename
    
    try:
        # 保存图片
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 读取图片数据
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # 获取目标尺寸、通道数与布局
        # 优先使用 adapter 自带的 input_layout（ONNXAdapter 自动检测），
        # 否则从 input_shape 启发式推断，兼容 NCHW/NHWC 两种布局。
        input_shape = model_info.get("input_shape", [1, 3, 224, 224])

        # adapter 已加载时，优先使用其 input_layout 属性（更准确）
        if adapter is not None and hasattr(adapter, "input_layout"):
            layout = adapter.input_layout
            # 从 input_shape 推断 target_size 和 target_channels（layout 已知）
            detected_layout, detected_size, detected_channels = _detect_layout_from_shape(input_shape)
            target_size = detected_size
            target_channels = detected_channels
        else:
            # 无 adapter 或未声明布局，自动检测
            layout, target_size, target_channels = _detect_layout_from_shape(input_shape)

        # 预处理（按检测到的布局输出 NCHW 或 NHWC，仅 /255 归一化，与训练一致）
        preprocessed = preprocess_image(image_data, target_size, target_channels, layout)

        # 执行推理：adapter 不为空时（示例模型或实验模型）走本地推理，否则走模型服务
        if adapter is not None:
            output, inference_time = adapter.infer(preprocessed)

            # 显式 softmax 后处理：根据 adapter.output_is_probability 决定
            # - SampleCNNAdapter: output_is_probability=True（内部已 softmax）→ 跳过
            # - PyTorchAdapter:   output_is_probability=False（返回 logits）→ 应用 softmax
            # - ONNXAdapter:      output_is_probability 取决于模型是否含 Softmax 节点
            # 替代原先"检测输出是否非负且和为1"的脆弱启发式，避免：
            #   1. LogSoftmax 输出被二次 softmax（概率的平方）
            #   2. logits 恰好全正且和接近 1 时错误跳过
            if not getattr(adapter, "output_is_probability", False):
                output_max = np.max(output, axis=-1, keepdims=True)
                exp_x = np.exp(output - output_max)
                output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

            # 构建预测结果
            output_flat = output.flatten()
            if len(output_flat) > 0:
                top_indices = np.argsort(output_flat)[-5:][::-1]
                predictions = [
                    {"class_id": int(idx), "probability": float(output_flat[idx]), "confidence": float(output_flat[idx])}
                    for idx in top_indices
                ]
            else:
                predictions = []
            
            # 提取激活值
            activations = {}
            try:
                all_acts = adapter.get_all_activations(preprocessed)
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
            except Exception as e:
                print(f"激活值提取失败: {e}")
            
            result = {
                "predictions": predictions,
                "inference_time": round(inference_time * 1000, 2),
                "layer_count": len(model_info.get("layers", [])),
                "activations": activations,
                "success": True,
                "input_size": list(preprocessed.shape[-2:]) if preprocessed.ndim >= 2 else [28, 28],
            }
        else:
            result = await model_service.inference(model_id, preprocessed)
        
        return {
            "code": 200,
            "message": "推理成功",
            "data": result
        }
        
    except ModelNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InferenceException as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        err_msg = str(e)
        # Bug13修复：将技术报错转换为用户可读提示
        if "INVALID_ARGUMENT" in err_msg and "shape" in err_msg.lower():
            raise HTTPException(
                status_code=422,
                detail="图片维度与模型输入不匹配，请检查模型期望的输入尺寸和通道数"
            )
        if "No module named" in err_msg:
            raise HTTPException(status_code=500, detail="模型依赖缺失，请联系管理员")
        raise HTTPException(status_code=500, detail=f"推理失败: {err_msg}")
    finally:
        await file.close()

@router.post("/gradcam")
async def gradcam_visualization(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    target_class: Optional[int] = Form(None),
    use_plusplus: bool = Form(True),
    alpha: float = Form(0.4),
):
    """
    Grad-CAM 显著性可视化接口
    对上传图片生成 Grad-CAM/Grad-CAM++ 热力图，叠加在原图上显示模型关注区域
    """
    import torch
    from app.ml.gradcam import generate_gradcam_visualization

    model_lower = model_id.lower().strip()
    is_sample = model_lower in ("sample_cnn", "sample", "default", "")

    adapter = None
    model_info = None
    pytorch_model = None

    if is_sample:
        try:
            from app.ml.factory import ModelAdapterFactory
            adapter = ModelAdapterFactory.create("sample_cnn")
            adapter.load_model("sample_cnn")
            pytorch_model = adapter.model
            model_info = {
                "model_id": "sample_cnn",
                "input_shape": [1, 1, 28, 28],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"示例模型加载失败: {str(e)}")
    else:
        exp_adapter, exp_model_info = _load_experiment_adapter(model_id)
        if exp_adapter is not None:
            adapter = exp_adapter
            model_info = exp_model_info
            pytorch_model = getattr(adapter, "model", None)
        else:
            raise HTTPException(status_code=404, detail="模型不存在或不支持Grad-CAM（仅支持训练好的PyTorch CNN模型）")

    if pytorch_model is None:
        raise HTTPException(status_code=400, detail="当前模型不支持Grad-CAM可视化（需要PyTorch CNN模型）")

    file_ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    image_data = await file.read()

    try:
        input_shape = model_info.get("input_shape", [1, 1, 28, 28])
        layout, target_size, target_channels = _detect_layout_from_shape(input_shape)
        preprocessed = preprocess_image(image_data, target_size, target_channels, layout)

        original_pil = Image.open(io.BytesIO(image_data))
        if target_channels == 1:
            if original_pil.mode != "L":
                original_pil = original_pil.convert("L")
        else:
            if original_pil.mode != "RGB":
                original_pil = original_pil.convert("RGB")
        original_pil = original_pil.resize(target_size, Image.BILINEAR)
        original_np = np.array(original_pil)

        input_tensor = torch.from_numpy(preprocessed).float()

        result = generate_gradcam_visualization(
            model=pytorch_model,
            input_tensor=input_tensor,
            original_image=original_np,
            target_class=target_class,
            use_plusplus=use_plusplus,
            alpha=alpha,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "code": 200,
            "message": "Grad-CAM生成成功",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM生成失败: {str(e)}")
    finally:
        await file.close()


@router.post("/batch")
async def batch_inference(
    model_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    批量图片推理
    """
    results = []
    
    for file in files:
        try:
            # 复用单张推理逻辑
            # 简化处理，实际应该提取为独立函数
            result = {"filename": file.filename, "status": "pending"}
            results.append(result)
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {
        "code": 200,
        "message": f"处理了 {len(files)} 张图片",
        "data": results
    }
