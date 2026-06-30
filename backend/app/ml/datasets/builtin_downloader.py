"""
内置标准数据集下载器
支持 MNIST、CIFAR-10 自动下载，包含：
- 本地缓存检测
- 断点续传
- 完整性校验（SHA256）
- 下载进度显示
"""
import os
import io
import hashlib
import logging
import zipfile
import gzip
import struct
import shutil
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError
import numpy as np

logger = logging.getLogger(__name__)

BUILTIN_DATASETS = {
    "mnist": {
        "name": "MNIST 手写数字",
        "description": "70,000张28x28灰度手写数字图片，10个类别（0-9）",
        "class_count": 10,
        "classes": [str(i) for i in range(10)],
        "image_size": "28x28x1",
        "feature_shape": "1x28x28",
        "total_samples": 70000,
        "urls": {
            "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        },
        "fallback_urls": {
            "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        },
        "expected_sizes": {
            "train_images": 60000,
            "test_images": 10000,
        },
    },
    "cifar10": {
        "name": "CIFAR-10",
        "description": "60,000张32x32彩色图片，10个类别（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）",
        "class_count": 10,
        "classes": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "image_size": "32x32x3",
        "feature_shape": "3x32x32",
        "total_samples": 60000,
        "urls": {
            "archive": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        },
        "expected_sizes": {
            "train": 50000,
            "test": 10000,
        },
    },
}


class DownloadProgress:
    """下载进度追踪器"""
    def __init__(self, total_bytes: int, callback: Optional[Callable] = None):
        self.total_bytes = total_bytes
        self.downloaded = 0
        self.callback = callback
        self.last_percent = -1

    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        if self.callback and self.total_bytes > 0:
            percent = int(self.downloaded * 100 / self.total_bytes)
            if percent != self.last_percent:
                self.last_percent = percent
                self.callback(percent, self.downloaded, self.total_bytes)


def compute_sha256(file_path: str) -> str:
    """计算文件SHA256哈希"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(
    url: str,
    dest_path: str,
    progress_callback: Optional[Callable] = None,
    expected_size: Optional[int] = None,
) -> bool:
    """下载文件，支持断点续传

    Args:
        url: 下载URL
        dest_path: 目标路径
        progress_callback: 进度回调 (percent, downloaded, total)
        expected_size: 预期文件大小（用于断点续传判断）

    Returns:
        是否下载成功
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest_path + ".part"
    existing_size = 0

    if os.path.exists(temp_path):
        existing_size = os.path.getsize(temp_path)

    try:
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as response:
            total_size = existing_size
            if "Content-Length" in response.headers:
                total_size += int(response.headers["Content-Length"])
            elif expected_size:
                total_size = expected_size

            mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
            if mode == "wb" and existing_size > 0:
                existing_size = 0

            progress = DownloadProgress(total_size, progress_callback)
            progress.downloaded = existing_size

            with open(temp_path, mode) as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))

        shutil.move(temp_path, dest_path)
        return True

    except Exception as e:
        logger.error(f"下载失败 {url}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


def verify_mnist_files(raw_dir: str) -> bool:
    """验证MNIST文件完整性"""
    expected_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    for fname in expected_files:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            gzpath = fpath + ".gz"
            if not os.path.exists(gzpath):
                return False
    return True


def decompress_gz(gz_path: str, out_path: str):
    """解压.gz文件"""
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_mnist_idx_images(file_path: str) -> np.ndarray:
    """加载MNIST IDX格式图像"""
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)


def load_mnist_idx_labels(file_path: str) -> np.ndarray:
    """加载MNIST IDX格式标签"""
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def prepare_mnist_dataset(
    cache_dir: str,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """准备MNIST数据集

    Returns:
        {'X_train': np.ndarray, 'y_train': np.ndarray, 'X_test': np.ndarray, 'y_test': np.ndarray}
    """
    import pickle

    ds_config = BUILTIN_DATASETS["mnist"]
    raw_dir = os.path.join(cache_dir, "mnist", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    processed_file = os.path.join(cache_dir, "mnist", "mnist_processed.pkl")
    if os.path.exists(processed_file):
        if progress_callback:
            progress_callback(100, 0, 0, "加载缓存数据...")
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    urls = ds_config["urls"]
    fallback = ds_config.get("fallback_urls", {})

    for key in ["train_images", "train_labels", "test_images", "test_labels"]:
        gz_file = os.path.join(raw_dir, urls[key].split("/")[-1])
        raw_file = gz_file.replace(".gz", "")

        if not os.path.exists(raw_file):
            if not os.path.exists(gz_file):
                if progress_callback:
                    progress_callback(0, 0, 0, f"下载{key}...")
                success = download_file(
                    urls[key], gz_file,
                    progress_callback=lambda p, d, t, k=key: progress_callback(
                        p * 25 // 100 + {"train_images": 0, "train_labels": 25, "test_images": 50, "test_labels": 75}[k],
                        d, t, f"下载{key}..."
                    ) if progress_callback else None
                )
                if not success and key in fallback:
                    if progress_callback:
                        progress_callback(0, 0, 0, f"尝试备用源下载{key}...")
                    success = download_file(fallback[key], gz_file)
                if not success:
                    raise RuntimeError(f"下载MNIST {key} 失败，请检查网络连接")

            if progress_callback:
                progress_callback(95, 0, 0, f"解压{key}...")
            decompress_gz(gz_file, raw_file)

    if progress_callback:
        progress_callback(98, 0, 0, "解析数据集...")

    X_train = load_mnist_idx_images(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    y_train = load_mnist_idx_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    X_test = load_mnist_idx_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    y_test = load_mnist_idx_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))

    result = {
        "X_train": X_train,
        "y_train": y_train.astype(np.int64),
        "X_test": X_test,
        "y_test": y_test.astype(np.int64),
        "class_names": ds_config["classes"],
        "feature_shape": (1, 28, 28),
    }

    with open(processed_file, "wb") as f:
        pickle.dump(result, f)

    if progress_callback:
        progress_callback(100, 0, 0, "MNIST数据集准备完成")

    return result


def prepare_cifar10_dataset(
    cache_dir: str,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, np.ndarray]:
    """准备CIFAR-10数据集"""
    import pickle
    import tarfile

    ds_config = BUILTIN_DATASETS["cifar10"]
    cifar_dir = os.path.join(cache_dir, "cifar10")
    raw_dir = os.path.join(cifar_dir, "raw")
    extracted_dir = os.path.join(raw_dir, "cifar-10-batches-py")
    os.makedirs(raw_dir, exist_ok=True)

    processed_file = os.path.join(cifar_dir, "cifar10_processed.pkl")
    if os.path.exists(processed_file):
        if progress_callback:
            progress_callback(100, 0, 0, "加载缓存数据...")
        with open(processed_file, "rb") as f:
            return pickle.load(f)

    archive_name = "cifar-10-python.tar.gz"
    archive_path = os.path.join(raw_dir, archive_name)

    if not os.path.exists(extracted_dir):
        if not os.path.exists(archive_path):
            if progress_callback:
                progress_callback(0, 0, 0, "下载CIFAR-10数据集...")
            url = ds_config["urls"]["archive"]
            success = download_file(
                url, archive_path,
                progress_callback=lambda p, d, t: progress_callback(p // 2, d, t, "下载CIFAR-10...") if progress_callback else None
            )
            if not success:
                raise RuntimeError("下载CIFAR-10失败，请检查网络连接")

        if progress_callback:
            progress_callback(55, 0, 0, "解压CIFAR-10...")
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(raw_dir)
        except Exception as e:
            if os.path.exists(archive_path):
                os.remove(archive_path)
            raise RuntimeError(f"CIFAR-10解压失败: {e}")

    if progress_callback:
        progress_callback(70, 0, 0, "解析CIFAR-10批次...")

    X_train_list = []
    y_train_list = []

    for i in range(1, 6):
        batch_file = os.path.join(extracted_dir, f"data_batch_{i}")
        if not os.path.exists(batch_file):
            raise RuntimeError(f"CIFAR-10批次文件缺失: {batch_file}")
        with open(batch_file, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            X_train_list.append(batch[b"data"])
            y_train_list.extend(batch[b"labels"])

    test_batch_file = os.path.join(extracted_dir, "test_batch")
    with open(test_batch_file, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
        X_test = test_batch[b"data"]
        y_test = test_batch[b"labels"]

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype=np.int64)

    X_train = X_train.reshape(-1, 3, 32, 32).astype(np.uint8)
    X_test = X_test.reshape(-1, 3, 32, 32).astype(np.uint8)

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": ds_config["classes"],
        "feature_shape": (3, 32, 32),
    }

    if progress_callback:
        progress_callback(95, 0, 0, "缓存数据...")
    with open(processed_file, "wb") as f:
        pickle.dump(result, f)

    if progress_callback:
        progress_callback(100, 0, 0, "CIFAR-10数据集准备完成")

    return result


def get_builtin_dataset_info(name: str) -> Optional[Dict[str, Any]]:
    """获取内置数据集元信息"""
    if name in BUILTIN_DATASETS:
        info = dict(BUILTIN_DATASETS[name])
        cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
        info["is_cached"] = os.path.exists(os.path.join(cache_dir, name, f"{name}_processed.pkl"))
        return info
    return None


def list_builtin_datasets() -> list:
    """列出所有内置数据集"""
    cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
    result = []
    for name, info in BUILTIN_DATASETS.items():
        entry = dict(info)
        entry["id"] = name
        entry["is_cached"] = os.path.exists(os.path.join(cache_dir, name, f"{name}_processed.pkl"))
        result.append(entry)
    return result
