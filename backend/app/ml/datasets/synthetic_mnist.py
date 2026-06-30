"""
合成 MNIST 数据集生成器
在无法下载真实 MNIST 时提供开箱即用的示例数据
生成类似手写数字的模式用于演示训练流程
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
from PIL import Image


def _draw_line(img, x0, y0, x1, y1, width=2, intensity=200):
    """简单的直线绘制"""
    h, w = img.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    points = []
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    for x, y in points:
        for px in range(max(0, x-width), min(w, x+width+1)):
            for py in range(max(0, y-width), min(h, y+width+1)):
                dist = np.sqrt((px-x)**2 + (py-y)**2)
                if dist <= width * 0.7:
                    val = int(intensity - dist*25)
                    if val > 0:
                        img[py, px] = min(255, int(img[py, px]) + val)


def _draw_circle(img, cx, cy, r, width=2, intensity=200):
    """绘制圆"""
    h, w = img.shape
    for a in np.linspace(0, 2*np.pi, 60):
        x = int(cx + r * np.cos(a))
        y = int(cy + r * np.sin(a))
        jx = np.random.randint(-1, 2)
        jy = np.random.randint(-1, 2)
        for px in range(max(0, x+jx-width), min(w, x+jx+width+1)):
            for py in range(max(0, y+jy-width), min(h, y+jy+width+1)):
                dist = np.sqrt((px-x-jx)**2 + (py-y-jy)**2)
                if dist <= width * 0.7:
                    val = int(intensity - dist*25)
                    if val > 0:
                        img[py, px] = min(255, int(img[py, px]) + val)


def _make_digit_pattern(digit, size=28):
    """生成单个数字的基础模式"""
    img = np.zeros((size, size), dtype=np.uint8)
    m = 4
    
    if digit == 0:
        _draw_circle(img, size//2, size//2, size//2-m-1, width=2)
    elif digit == 1:
        _draw_line(img, size//2, m, size//2, size-m-1, width=2)
        _draw_line(img, size//2-2, m+2, size//2, m, width=1)
    elif digit == 2:
        _draw_circle(img, size//2, m+size//4, size//4-1, width=2)
        _draw_line(img, size//2+size//4-1, m+size//4+2, m+1, size-m-2, width=2)
        _draw_line(img, m, size-m-1, size-m-1, size-m-1, width=2)
    elif digit == 3:
        _draw_circle(img, size//2, m+size//4, size//4-2, width=2)
        _draw_circle(img, size//2, size-m-size//4, size//4-2, width=2)
    elif digit == 4:
        _draw_line(img, size-m-1, m, size-m-1, size-m-1, width=2)
        _draw_line(img, m, size//2, size-m-1, size//2, width=2)
        _draw_line(img, m, m, m, size//2, width=2)
    elif digit == 5:
        _draw_line(img, size-m-1, m+2, m+1, m+2, width=2)
        _draw_line(img, m, m+2, m, size//2, width=2)
        _draw_circle(img, size//2, size-m-size//4, size//4-2, width=2)
        _draw_line(img, size-m-1, size//2-1, size-m-1, size-m-1, width=2)
    elif digit == 6:
        _draw_circle(img, size//2, size//2+2, size//2-m-1, width=2)
        _draw_line(img, size-m-1, m+2, size-m-1, size//2, width=2)
    elif digit == 7:
        _draw_line(img, m, m+2, size-m-1, m+2, width=2)
        _draw_line(img, size-m-1, m+2, size//2, size-m-1, width=2)
    elif digit == 8:
        _draw_circle(img, size//2, m+size//4, size//4-2, width=2)
        _draw_circle(img, size//2, size-m-size//4, size//4-2, width=2)
    elif digit == 9:
        _draw_circle(img, size//2, m+size//4, size//4-2, width=2)
        _draw_line(img, m, size-m-1, m, m+size//4-2, width=2)
        _draw_line(img, size-m-1, m+size//4, size-m-1, size-m-1, width=2)
    
    noise = np.random.randint(0, 20, (size, size), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 8, 0, 255).astype(np.uint8)
    return img


def _augment_digit(img, size=28):
    """使用PIL进行简单数据增强"""
    pil = Image.fromarray(img, mode='L')
    
    angle = float(np.random.uniform(-10, 10))
    pil = pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    
    import random
    sx = random.randint(-2, 2)
    sy = random.randint(-2, 2)
    scale = float(np.random.uniform(0.88, 1.12))
    
    new_w = int(size * scale)
    new_h = int(size * scale)
    if new_w > 0 and new_h > 0:
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
    
    canvas = Image.new('L', (size, size), 0)
    paste_x = (size - new_w) // 2 + sx
    paste_y = (size - new_h) // 2 + sy
    canvas.paste(pil, (paste_x, paste_y))
    
    return np.array(canvas, dtype=np.uint8)


def generate_synthetic_mnist(
    n_train=2000,
    n_test=500,
    seed=42,
):
    """生成合成 MNIST 数据集"""
    np.random.seed(seed)
    
    def _gen_split(n):
        X, y = [], []
        per_class = n // 10
        remainder = n % 10
        for digit in range(10):
            count = per_class + (1 if digit < remainder else 0)
            for _ in range(count):
                img = _make_digit_pattern(digit)
                try:
                    img = _augment_digit(img)
                except Exception:
                    pass
                X.append(img)
                y.append(digit)
        indices = np.random.permutation(len(X))
        return np.array(X)[indices], np.array(y, dtype=np.int64)[indices]
    
    X_train, y_train = _gen_split(n_train)
    X_test, y_test = _gen_split(n_test)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": [str(i) for i in range(10)],
        "feature_shape": (1, 28, 28),
    }


def ensure_sample_dataset(cache_dir=None, n_train=2000, n_test=500):
    """确保示例数据集存在，不存在则生成"""
    if cache_dir is None:
        cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
    
    sample_dir = os.path.join(cache_dir, "sample_mnist")
    os.makedirs(sample_dir, exist_ok=True)
    processed_file = os.path.join(sample_dir, "sample_mnist_processed.pkl")
    
    if os.path.exists(processed_file):
        return processed_file
    
    data = generate_synthetic_mnist(n_train, n_test)
    with open(processed_file, "wb") as f:
        pickle.dump(data, f)
    
    return processed_file


def load_sample_dataset(cache_dir=None, split="train"):
    """加载示例数据集（合成MNIST）"""
    processed_file = ensure_sample_dataset(cache_dir)
    
    with open(processed_file, "rb") as f:
        data = pickle.load(f)
    
    if split == "test":
        X, y = data["X_test"], data["y_test"]
    else:
        X, y = data["X_train"], data["y_train"]
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "mnist_idx",
        "sample_count": X.shape[0],
        "feature_shape": "1x28x28",
        "dataset_name": "SampleMNIST (示例)",
        "class_names": data.get("class_names"),
    }
