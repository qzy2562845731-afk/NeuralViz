"""
pytest 配置文件 - 共享 fixtures 和测试工具
"""
import sys
import os
import pytest
import torch
import numpy as np

# 确保 backend 目录在 Python path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_image_data():
    """生成模拟图像数据 (MNIST-like)"""
    np.random.seed(42)
    X = np.random.rand(100, 1, 28, 28).astype(np.float32)
    y = np.random.randint(0, 10, 100).astype(np.int64)
    return X, y


@pytest.fixture
def sample_tabular_data():
    """生成模拟表格数据"""
    np.random.seed(42)
    X = np.random.rand(100, 64).astype(np.float32)
    y = np.random.randint(0, 5, 100).astype(np.int64)
    return X, y


@pytest.fixture
def sample_image_tensors(sample_image_data):
    """生成图像数据 PyTorch 张量"""
    X, y = sample_image_data
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor


@pytest.fixture
def sample_tabular_tensors(sample_tabular_data):
    """生成表格数据 PyTorch 张量"""
    X, y = sample_tabular_data
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return X_tensor, y_tensor