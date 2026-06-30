"""
安全工具模块
提供路径安全校验、输入净化等通用安全功能
"""
import re
import os
from pathlib import Path
from typing import Optional


# 合法的 ID 字符集：仅允许 UUID 格式 (a-f0-9-) 和通用字母数字下划线连字符
_VALID_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')
# 最大 ID 长度，防止 DoS 攻击
_MAX_ID_LENGTH = 256


def sanitize_path_id(id_value: str, max_length: int = _MAX_ID_LENGTH) -> str:
    """净化用于文件路径构造的 ID 参数，防止路径穿越攻击

    规则：
    1. 长度限制（默认 256 字符）
    2. 仅允许字母数字、下划线、连字符、点号
    3. 禁止路径分隔符（/、\\）和路径穿越序列（..）
    4. 禁止空字符串

    Args:
        id_value: 待校验的 ID 字符串
        max_length: 最大允许长度

    Returns:
        净化后的 ID 字符串

    Raises:
        ValueError: 输入不符合安全规则
    """
    if not id_value or not id_value.strip():
        raise ValueError("ID 不能为空")

    if len(id_value) > max_length:
        raise ValueError(f"ID 长度超过限制 ({max_length})")

    # 禁止路径穿越字符
    if '..' in id_value:
        raise ValueError("ID 包含非法路径穿越序列: '..'")

    if '/' in id_value or '\\' in id_value:
        raise ValueError("ID 包含非法路径分隔符")

    # 仅允许安全字符集
    if not _VALID_ID_PATTERN.match(id_value):
        raise ValueError(f"ID 包含非法字符: '{id_value}'")

    return id_value.strip()


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """净化文件名，移除潜在危险字符

    Args:
        filename: 原始文件名
        max_length: 最大允许长度

    Returns:
        净化后的安全文件名
    """
    if not filename:
        return "unnamed"

    # 提取纯文件名（去掉路径）
    basename = os.path.basename(filename)

    # 移除危险字符，仅保留安全字符
    safe = re.sub(r'[^\w\.\-]', '_', basename)

    # 移除连续下划线
    safe = re.sub(r'_+', '_', safe)

    # 去掉首尾的下划线和点号
    safe = safe.strip('_.')

    if not safe:
        return "unnamed"

    if len(safe) > max_length:
        name, ext = os.path.splitext(safe)
        safe = name[:max_length - len(ext)] + ext

    return safe


def safe_path_join(base_dir: Path, *paths: str) -> Path:
    """安全地拼接路径，防止路径穿越

    Args:
        base_dir: 基础目录（绝对路径）
        *paths: 要拼接的路径片段

    Returns:
        安全的拼接路径

    Raises:
        ValueError: 检测到路径穿越攻击
    """
    base_dir = base_dir.resolve()
    result = base_dir.joinpath(*paths).resolve()

    # 确保结果路径在 base_dir 内
    try:
        result.relative_to(base_dir)
    except ValueError:
        raise ValueError(f"检测到路径穿越攻击: {paths}")

    return result