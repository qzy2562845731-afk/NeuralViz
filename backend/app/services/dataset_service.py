"""
数据集服务层
处理数据集上传、解压、解析、版本管理、CRUD
使用插件化解析器架构，自动识别数据集格式
"""
import os
import json
import uuid
import hashlib
import zipfile
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

from app.core.security import sanitize_path_id
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.config import settings
from app.models.dataset import Dataset
from app.ml.datasets.factory import DatasetParserFactory
from app.ml.datasets.base_parser import BaseDatasetParser, DatasetParseError, UnrecognizedFormatError

logger = logging.getLogger(__name__)

# 最大上传文件大小：500MB
_MAX_FILE_SIZE = 500 * 1024 * 1024


class DatasetService:
    """数据集服务"""

    _instance: Optional['DatasetService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache: Dict[str, Dict] = {}

    def _get_db(self) -> Session:
        return SessionLocal()

    def _to_dict(self, ds: Dataset) -> Dict[str, Any]:
        return {
            "dataset_id": ds.dataset_id,
            "name": ds.name,
            "description": ds.description or "",
            "version": ds.version,
            "file_path": ds.file_path,
            "extract_path": ds.extract_path,
            "sample_count": ds.sample_count,
            "class_count": ds.class_count,
            "image_size": ds.image_size,
            "feature_shape": ds.feature_shape,
            "dataset_type": ds.dataset_type,
            "class_distribution": json.loads(ds.class_distribution) if ds.class_distribution else {},
            "file_hash": ds.file_hash,
            "status": ds.status,
            "error_message": ds.error_message,
            "tags": ds.tags.split(",") if ds.tags else [],
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
            "updated_at": ds.updated_at.isoformat() if ds.updated_at else None,
        }

    # ============================================================
    # 数据集上传
    # ============================================================

    def upload_dataset(
        self,
        file_content: bytes,
        filename: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """上传数据集压缩包

        - 保存文件到 uploads/datasets/
        - 计算 SHA256 哈希，同哈希同名称判定为同一版本，避免重复存储
        - 同名数据集新文件自动生成新版本号（v1、v2...）
        - 创建数据库记录，状态为 processing
        - 异步启动解析流程
        """
        # 大小校验
        if len(file_content) > _MAX_FILE_SIZE:
            raise ValueError(f"文件大小超过限制（最大 {_MAX_FILE_SIZE // (1024*1024)}MB）")

        # 计算哈希
        file_hash = hashlib.sha256(file_content).hexdigest()

        # 从文件名提取数据集名称（如果未提供 name）
        if not name:
            name = Path(filename).stem

        # 检查同哈希同名称是否已存在（去重）
        db = self._get_db()
        try:
            existing = db.query(Dataset).filter(
                Dataset.file_hash == file_hash,
                Dataset.name == name,
                Dataset.is_deleted == False,
            ).first()
            if existing:
                return self._to_dict(existing)

            # 生成版本号：同名数据集自动递增
            version = self._next_version(db, name)

            # 生成 dataset_id
            dataset_id = str(uuid.uuid4())

            # 保存文件
            settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
            file_ext = Path(filename).suffix.lower()
            saved_filename = f"{dataset_id}{file_ext}"
            saved_path = settings.DATASET_DIR / saved_filename

            with open(saved_path, "wb") as f:
                f.write(file_content)

            # 创建数据库记录
            ds = Dataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                version=version,
                file_path=str(saved_path),
                file_hash=file_hash,
                status="processing",
                tags=",".join(tags) if tags else "",
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)

            result = self._to_dict(ds)
            self._cache[dataset_id] = result

            # 异步解析
            thread = threading.Thread(
                target=self._process_dataset_async,
                args=(dataset_id, str(saved_path)),
                daemon=True,
            )
            thread.start()

            return result
        finally:
            db.close()

    def _next_version(self, db: Session, name: str) -> str:
        """生成下一个版本号：同名数据集中最大版本号 +1"""
        existing = db.query(Dataset).filter(
            Dataset.name == name,
            Dataset.is_deleted == False,
        ).all()

        if not existing:
            return "v1"

        max_num = 0
        for ds in existing:
            v = ds.version or "v1"
            try:
                num = int(v.lstrip("vV"))
                if num > max_num:
                    max_num = num
            except ValueError:
                pass

        return f"v{max_num + 1}"

    # ============================================================
    # 异步解析（插件化解析器架构）
    # ============================================================

    def _process_dataset_async(self, dataset_id: str, zip_path: str):
        """异步解析数据集：解压 → 工厂识别格式 → 解析器解析元数据"""
        dataset_id = sanitize_path_id(dataset_id)
        extract_path = settings.DATASET_DIR / f"{dataset_id}_extracted"

        db = self._get_db()
        try:
            ds = db.query(Dataset).filter(
                Dataset.dataset_id == dataset_id,
            ).first()
            if not ds:
                return

            # Step 1: 解压
            try:
                logger.info(f"[数据集 {dataset_id}] 开始解压: {zip_path}")
                self._safe_extract_zip(zip_path, str(extract_path))
                logger.info(f"[数据集 {dataset_id}] 解压完成: {extract_path}")
            except Exception as e:
                ds.status = "failed"
                ds.error_message = f"解压失败: {str(e)}"
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return

            # Step 2: 工厂自动识别格式
            try:
                parser = DatasetParserFactory.detect_and_create(str(extract_path))
                logger.info(f"[数据集 {dataset_id}] 识别到格式: {parser.dataset_type}")
            except UnrecognizedFormatError as e:
                ds.status = "failed"
                ds.error_message = str(e)
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return
            except Exception as e:
                ds.status = "failed"
                ds.error_message = f"格式识别失败: {str(e)}"
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return

            # Step 3: 解析元数据
            try:
                stats = parser.parse_metadata(str(extract_path))
                logger.info(
                    f"[数据集 {dataset_id}] 解析完成: "
                    f"类型={stats['dataset_type']}, "
                    f"样本数={stats['sample_count']}, "
                    f"类别数={stats['class_count']}, "
                    f"特征维度={stats['feature_shape']}"
                )
            except DatasetParseError as e:
                ds.status = "failed"
                ds.error_message = str(e)
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return
            except Exception as e:
                ds.status = "failed"
                ds.error_message = f"解析失败: {str(e)}"
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return

            # Step 4: 更新记录
            ds.extract_path = str(extract_path)
            ds.sample_count = stats["sample_count"]
            ds.class_count = stats["class_count"]
            ds.image_size = stats["feature_shape"]  # 向后兼容：image_size 字段也存储特征维度
            ds.feature_shape = stats["feature_shape"]
            ds.dataset_type = stats["dataset_type"]
            ds.class_distribution = json.dumps(stats["class_distribution"])
            ds.status = "ready"
            ds.error_message = None
            ds.updated_at = datetime.utcnow()
            db.commit()

            self._cache[dataset_id] = self._to_dict(ds)
        except Exception as e:
            # 兜底：标记失败
            logger.error(f"[数据集 {dataset_id}] 未知错误: {str(e)}", exc_info=True)
            try:
                ds = db.query(Dataset).filter(
                    Dataset.dataset_id == dataset_id,
                ).first()
                if ds:
                    ds.status = "failed"
                    ds.error_message = f"未知错误: {str(e)}"
                    ds.updated_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            self._cache.pop(dataset_id, None)
        finally:
            db.close()

    def _safe_extract_zip(self, zip_path: str, extract_dir: str):
        """安全解压 zip 文件，防止路径遍历漏洞"""
        os.makedirs(extract_dir, exist_ok=True)
        extract_dir = os.path.abspath(extract_dir)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                member_path = os.path.abspath(os.path.join(extract_dir, member))
                # 安全校验：确保解压路径在目标目录内
                if not member_path.startswith(extract_dir + os.sep) and member_path != extract_dir:
                    raise ValueError(f"检测到路径遍历攻击: {member}")
            zf.extractall(extract_dir)

    # ============================================================
    # 重新解析
    # ============================================================

    def reparse_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """重新解析失败的数据集

        - 清除旧的解压目录（如果存在）
        - 重新解压并解析
        - 状态流转：failed → processing → ready / failed
        """
        db = self._get_db()
        try:
            ds = db.query(Dataset).filter(
                Dataset.dataset_id == dataset_id,
                Dataset.is_deleted == False,
            ).first()
            if not ds:
                return None

            # 更新状态为 processing
            ds.status = "processing"
            ds.error_message = None
            ds.updated_at = datetime.utcnow()
            db.commit()

            result = self._to_dict(ds)
            self._cache.pop(dataset_id, None)

            # 异步重新解析
            thread = threading.Thread(
                target=self._process_dataset_async,
                args=(dataset_id, ds.file_path),
                daemon=True,
            )
            thread.start()

            return result
        finally:
            db.close()

    # ============================================================
    # 数据加载（供训练引擎调用）
    # ============================================================

    def cleanup_invalid_records(self) -> int:
        """清理数据库中文件已不存在的无效记录，返回清理数量"""
        db = self._get_db()
        deleted = 0
        try:
            all_ds = db.query(Dataset).filter(Dataset.is_deleted == False).all()
            for ds in all_ds:
                fp = ds.file_path or ""
                ep = ds.extract_path or ""
                if fp.startswith("builtin://"):
                    continue
                invalid = False
                if fp and not fp.startswith("builtin://") and not os.path.exists(fp):
                    invalid = True
                if ep and not os.path.exists(ep):
                    invalid = True
                if invalid:
                    ds.is_deleted = True
                    deleted += 1
                    logger.warning(f"清理无效数据集记录: {ds.name} ({ds.dataset_id}) - 文件不存在")
            if deleted > 0:
                db.commit()
        except Exception as e:
            logger.error(f"清理无效记录失败: {e}")
            db.rollback()
        finally:
            db.close()
        self._cache.clear()
        return deleted

    def ensure_sample_dataset(self) -> Dict[str, Any]:
        """确保示例数据集（SampleMNIST）存在，不存在则创建并注册到数据库
        
        Returns:
            示例数据集的信息字典
        """
        from app.ml.datasets.synthetic_mnist import ensure_sample_dataset as _ensure_sample, load_sample_dataset
        
        cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")
        _ensure_sample(cache_dir)
        sample_dir = os.path.join(cache_dir, "sample_mnist")
        
        db = self._get_db()
        try:
            existing = db.query(Dataset).filter(
                Dataset.file_path == "builtin://sample_mnist",
                Dataset.is_deleted == False,
            ).first()
            
            if existing and existing.status == "ready":
                return self._to_dict(existing)
            
            import pickle as pkl
            processed_file = os.path.join(sample_dir, "sample_mnist_processed.pkl")
            with open(processed_file, "rb") as f:
                data = pkl.load(f)
            
            y = data["y_train"]
            class_dist = {}
            for label in y:
                l = str(int(label))
                class_dist[l] = class_dist.get(l, 0) + 1
            
            if existing:
                existing.status = "ready"
                existing.extract_path = sample_dir
                existing.sample_count = len(y)
                existing.class_count = 10
                existing.feature_shape = "1x28x28"
                existing.dataset_type = "mnist_idx"
                existing.class_distribution = json.dumps(class_dist)
                ds = existing
            else:
                ds_id = str(uuid.uuid4())
                ds = Dataset(
                    dataset_id=ds_id,
                    name="SampleMNIST (示例)",
                    description="内置合成手写数字示例数据集，开箱即用，无需下载",
                    version="v1",
                    file_path="builtin://sample_mnist",
                    extract_path=sample_dir,
                    sample_count=len(y),
                    class_count=10,
                    image_size="28x28x1",
                    feature_shape="1x28x28",
                    dataset_type="mnist_idx",
                    class_distribution=json.dumps(class_dist),
                    status="ready",
                    tags="builtin,sample,example,mnist",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                db.add(ds)
            
            db.commit()
            self._cache.clear()
            return self._to_dict(ds)
        finally:
            db.close()

    def load_dataset(self, dataset_id: str, split: str = "train") -> Dict[str, Any]:
        """加载数据集数据

        Args:
            dataset_id: 数据集ID
            split: 'train' 或 'test'

        Returns:
            dict with keys:
                - X: 特征数组
                - y: 标签数组
                - dataset_type: 格式类型
                - sample_count: 样本数
                - feature_shape: 特征维度
                - class_names: 类别名称列表（如果有）
        """
        ds_info = self.get_dataset(dataset_id)
        if not ds_info:
            raise ValueError("数据集不存在")

        if ds_info["status"] != "ready":
            raise ValueError(f"数据集未就绪，当前状态: {ds_info['status']}")

        file_path = ds_info.get("file_path", "")
        extract_path = ds_info.get("extract_path", "")

        if file_path and file_path.startswith("builtin://"):
            return self._load_builtin_dataset(file_path, ds_info, split)

        if not extract_path or not os.path.exists(extract_path):
            self.cleanup_invalid_records()
            raise ValueError("解压目录不存在，已自动清理无效记录，请重新选择数据集")

        parser = DatasetParserFactory.detect_and_create(extract_path)
        X, y = parser.load_data(extract_path, split=split)

        return {
            "X": X,
            "y": y,
            "dataset_type": parser.dataset_type,
            "sample_count": X.shape[0],
            "feature_shape": ds_info.get("feature_shape"),
            "dataset_name": ds_info.get("name", ""),
            "class_names": None,
        }

    def _load_builtin_dataset(self, file_path: str, ds_info: Dict, split: str) -> Dict[str, Any]:
        """加载内置数据集（MNIST/CIFAR-10/SampleMNIST）"""
        import pickle
        from pathlib import Path

        dataset_name = file_path.replace("builtin://", "")
        cache_dir = os.path.join(str(Path.home()), ".neuralviz", "datasets")

        if dataset_name == "sample_mnist":
            from app.ml.datasets.synthetic_mnist import load_sample_dataset
            return load_sample_dataset(cache_dir, split)
        elif dataset_name == "mnist":
            processed_file = os.path.join(cache_dir, "mnist", "mnist_processed.pkl")
            dataset_type = "mnist_idx"
        elif dataset_name == "cifar10":
            processed_file = os.path.join(cache_dir, "cifar10", "cifar10_processed.pkl")
            dataset_type = "image_folder"
        else:
            raise ValueError(f"不支持的内置数据集: {dataset_name}")

        if dataset_name != "sample_mnist" and not os.path.exists(processed_file):
            if dataset_name == "mnist":
                from app.ml.datasets.synthetic_mnist import load_sample_dataset
                result = load_sample_dataset(cache_dir, split)
                result["dataset_name"] = ds_info.get("name", "MNIST")
                return result
            raise ValueError(f"内置数据集 {dataset_name} 尚未下载，请先下载")

        with open(processed_file, "rb") as f:
            data = pickle.load(f)

        if split == "test":
            X, y = data["X_test"], data["y_test"]
        else:
            X, y = data["X_train"], data["y_train"]

        return {
            "X": X,
            "y": y,
            "dataset_type": dataset_type,
            "sample_count": X.shape[0],
            "feature_shape": ds_info.get("feature_shape"),
            "dataset_name": ds_info.get("name", ""),
            "class_names": data.get("class_names"),
        }

    # ============================================================
    # CRUD
    # ============================================================

    def list_datasets(
        self,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        tags: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """分页查询数据集列表"""
        db = self._get_db()
        try:
            query = db.query(Dataset).filter(Dataset.is_deleted == False)

            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    Dataset.name.like(search_pattern)
                    | Dataset.description.like(search_pattern)
                )

            if tags:
                for tag in tags.split(","):
                    tag = tag.strip()
                    if tag:
                        query = query.filter(Dataset.tags.like(f"%{tag}%"))

            if status:
                query = query.filter(Dataset.status == status)

            total = query.count()

            datasets = (
                query.order_by(Dataset.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )

            items = [self._to_dict(ds) for ds in datasets]

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }
        finally:
            db.close()

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集详情"""
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        db = self._get_db()
        try:
            ds = db.query(Dataset).filter(
                Dataset.dataset_id == dataset_id,
                Dataset.is_deleted == False,
            ).first()
            if ds:
                result = self._to_dict(ds)
                self._cache[dataset_id] = result
                return result
            return None
        finally:
            db.close()

    def get_versions(self, name: str) -> List[Dict[str, Any]]:
        """获取同名数据集的所有版本列表"""
        db = self._get_db()
        try:
            datasets = db.query(Dataset).filter(
                Dataset.name == name,
                Dataset.is_deleted == False,
            ).order_by(Dataset.version.asc()).all()

            return [self._to_dict(ds) for ds in datasets]
        finally:
            db.close()

    def delete_dataset(self, dataset_id: str) -> bool:
        """软删除数据集"""
        db = self._get_db()
        try:
            ds = db.query(Dataset).filter(
                Dataset.dataset_id == dataset_id,
                Dataset.is_deleted == False,
            ).first()

            if not ds:
                return False

            ds.is_deleted = True
            ds.updated_at = datetime.utcnow()
            db.commit()

            self._cache.pop(dataset_id, None)
            return True
        finally:
            db.close()

    # ============================================================
    # 直接文件上传（非 zip 格式：CSV / JSON / NumPy）
    # ============================================================

    def upload_direct_dataset(
        self,
        file_content: bytes,
        filename: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """直接上传非压缩格式的数据集文件（CSV / JSON / NumPy）

        与 upload_dataset 不同，该方法不经过解压流程，
        而是直接将文件保存到 extract_path 目录并调用解析器解析。
        """
        if len(file_content) > _MAX_FILE_SIZE:
            raise ValueError(f"文件大小超过限制（最大 {_MAX_FILE_SIZE // (1024*1024)}MB）")

        file_hash = hashlib.sha256(file_content).hexdigest()
        if not name:
            name = Path(filename).stem

        file_ext = Path(filename).suffix.lower()
        supported = {".csv", ".tsv", ".json", ".npy", ".npz"}
        if file_ext not in supported:
            raise ValueError(f"不支持的文件格式: {file_ext}，支持: {', '.join(sorted(supported))}")

        # 去重检查
        db = self._get_db()
        try:
            existing = db.query(Dataset).filter(
                Dataset.file_hash == file_hash,
                Dataset.name == name,
                Dataset.is_deleted == False,
            ).first()
            if existing:
                return self._to_dict(existing)

            version = self._next_version(db, name)
            dataset_id = str(uuid.uuid4())

            # 保存文件到专用目录
            settings.DATASET_DIR.mkdir(parents=True, exist_ok=True)
            extract_dir = settings.DATASET_DIR / f"{dataset_id}_extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            saved_path = extract_dir / filename
            with open(saved_path, "wb") as f:
                f.write(file_content)

            # 创建数据库记录
            ds = Dataset(
                dataset_id=dataset_id,
                name=name,
                description=description,
                version=version,
                file_path=str(saved_path),
                extract_path=str(extract_dir),
                file_hash=file_hash,
                status="processing",
                tags=",".join(tags) if tags else "",
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)

            result = self._to_dict(ds)
            self._cache[dataset_id] = result

            # 同步解析（非压缩文件解析很快，无需异步）
            thread = threading.Thread(
                target=self._process_direct_dataset,
                args=(dataset_id, str(extract_dir)),
                daemon=True,
            )
            thread.start()

            return result
        finally:
            db.close()

    def _process_direct_dataset(self, dataset_id: str, extract_path: str):
        """处理直接上传的数据集文件（CSV/JSON/NumPy）"""
        dataset_id = sanitize_path_id(dataset_id)
        db = self._get_db()
        try:
            ds = db.query(Dataset).filter(
                Dataset.dataset_id == dataset_id,
            ).first()
            if not ds:
                return

            try:
                parser = DatasetParserFactory.detect_and_create(str(extract_path))
                logger.info(f"[数据集 {dataset_id}] 识别到格式: {parser.dataset_type}")
            except UnrecognizedFormatError as e:
                ds.status = "failed"
                ds.error_message = str(e)
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return
            except Exception as e:
                ds.status = "failed"
                ds.error_message = f"格式识别失败: {str(e)}"
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return

            try:
                stats = parser.parse_metadata(str(extract_path))
                logger.info(
                    f"[数据集 {dataset_id}] 解析完成: "
                    f"类型={stats['dataset_type']}, "
                    f"样本数={stats['sample_count']}, "
                    f"类别数={stats['class_count']}, "
                    f"特征维度={stats['feature_shape']}"
                )
            except DatasetParseError as e:
                ds.status = "failed"
                ds.error_message = str(e)
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return
            except Exception as e:
                ds.status = "failed"
                ds.error_message = f"解析失败: {str(e)}"
                ds.updated_at = datetime.utcnow()
                db.commit()
                self._cache.pop(dataset_id, None)
                return

            ds.sample_count = stats["sample_count"]
            ds.class_count = stats["class_count"]
            ds.image_size = stats["feature_shape"]
            ds.feature_shape = stats["feature_shape"]
            ds.dataset_type = stats["dataset_type"]
            ds.class_distribution = json.dumps(stats["class_distribution"])
            ds.status = "ready"
            ds.error_message = None
            ds.updated_at = datetime.utcnow()
            db.commit()

            self._cache[dataset_id] = self._to_dict(ds)
        except Exception as e:
            logger.error(f"[数据集 {dataset_id}] 未知错误: {str(e)}", exc_info=True)
            try:
                ds = db.query(Dataset).filter(
                    Dataset.dataset_id == dataset_id,
                ).first()
                if ds:
                    ds.status = "failed"
                    ds.error_message = f"未知错误: {str(e)}"
                    ds.updated_at = datetime.utcnow()
                    db.commit()
            except Exception:
                pass
            self._cache.pop(dataset_id, None)
        finally:
            db.close()

    # ============================================================
    # 工具方法（供训练模块调用）
    # ============================================================

    def get_dataset_path(self, dataset_id: str) -> Optional[str]:
        """根据 dataset_id 获取数据集解压后的目录路径"""
        ds_info = self.get_dataset(dataset_id)
        if not ds_info:
            return None
        return ds_info.get("extract_path")

    def verify_integrity(self, dataset_id: str) -> Dict[str, Any]:
        """校验数据集完整性

        检查项：
        - 解压目录是否存在
        - 样本数是否与记录一致
        - 类别数是否与记录一致
        """
        ds_info = self.get_dataset(dataset_id)
        if not ds_info:
            return {"valid": False, "reason": "数据集不存在"}

        extract_path = ds_info.get("extract_path")
        if not extract_path or not os.path.exists(extract_path):
            return {"valid": False, "reason": "解压目录不存在"}

        # 使用解析器重新统计
        try:
            parser = DatasetParserFactory.detect_and_create(extract_path)
            stats = parser.parse_metadata(extract_path)
        except Exception as e:
            return {"valid": False, "reason": f"解析失败: {str(e)}"}

        sample_match = stats["sample_count"] == ds_info["sample_count"]
        class_match = stats["class_count"] == ds_info["class_count"]

        return {
            "valid": sample_match and class_match,
            "sample_count": stats["sample_count"],
            "expected_sample_count": ds_info["sample_count"],
            "class_count": stats["class_count"],
            "expected_class_count": ds_info["class_count"],
            "sample_match": sample_match,
            "class_match": class_match,
        }
