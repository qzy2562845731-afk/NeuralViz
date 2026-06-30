"""
结构化JSON日志系统
- 支持控制台输出（彩色）和文件输出（JSON格式）
- 包含时间戳、模块、级别、用户ID等上下文信息
- 兼容ELK等日志分析平台
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# 日志目录
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class JsonFormatter(logging.Formatter):
    """JSON格式化器，输出结构化日志"""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # 额外字段（如 user_id, experiment_id, task_id 等）
        if self.include_extra:
            for key in ("user_id", "experiment_id", "task_id", "dataset_id", "model_id", "duration_ms", "epoch"):
                val = getattr(record, key, None)
                if val is not None:
                    log_entry[key] = val

        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """彩色控制台格式化器（开发环境使用）"""

    COLORS = {
        "DEBUG": "\033[36m",     # 青色
        "INFO": "\033[32m",      # 绿色
        "WARNING": "\033[33m",   # 黄色
        "ERROR": "\033[31m",     # 红色
        "CRITICAL": "\033[35m",  # 紫色
    }
    RESET = "\033[0m"
    GRAY = "\033[90m"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:<8}{self.RESET}"
        module = f"{self.GRAY}{record.module}:{record.lineno}{self.RESET}"
        message = record.getMessage()

        # 额外上下文
        extra_parts = []
        for key in ("experiment_id", "epoch"):
            val = getattr(record, key, None)
            if val is not None:
                extra_parts.append(f"{self.GRAY}{key}={val}{self.RESET}")

        extra_str = " " + " ".join(extra_parts) if extra_parts else ""
        return f"{self.GRAY}{timestamp}{self.RESET} {level} {module}{extra_str} {message}"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    json_file: bool = True,
) -> None:
    """初始化日志系统

    Args:
        level: 日志级别
        log_file: 日志文件路径（None则使用默认路径）
        console: 是否启用控制台输出（彩色）
        json_file: 是否启用JSON文件输出
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除已有处理器
    root_logger.handlers.clear()

    # 控制台处理器（彩色）
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredConsoleFormatter())
        root_logger.addHandler(console_handler)

    # JSON文件处理器
    if json_file:
        if log_file is None:
            log_file = str(LOG_DIR / f"neuralviz_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    # 降低第三方库日志级别
    for lib in ("uvicorn", "uvicorn.access", "sqlalchemy", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取命名日志器

    Usage:
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("训练开始", extra={"experiment_id": "xxx", "epoch": 1})
    """
    return logging.getLogger(name)


class LogContext:
    """日志上下文管理器，自动注入 experiment_id 等字段

    Usage:
        with LogContext(experiment_id="exp_123"):
            logger.info("处理中")  # 自动附带 experiment_id
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._old_factory = None

    def __enter__(self):
        self._old_factory = logging.getLogRecordFactory()
        extra = self.kwargs

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            for key, val in extra.items():
                setattr(record, key, val)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)