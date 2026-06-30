from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
from .config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=settings.DEBUG
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db():
    """初始化数据库表"""
    from app.models import model_metadata, experiment, experiment_metric, dataset  # noqa
    Base.metadata.create_all(bind=engine)
    _migrate_datasets_table()
    _migrate_experiment_metrics_table()


def _migrate_datasets_table():
    """数据集表迁移：为已有表添加新字段（SQLite ALTER TABLE）"""
    from sqlalchemy import text, inspect
    inspector = inspect(engine)

    if "datasets" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("datasets")}
    new_columns = [
        ("dataset_type", "VARCHAR(32)"),
        ("feature_shape", "VARCHAR(64)"),
    ]

    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                conn.execute(text(f"ALTER TABLE datasets ADD COLUMN {col_name} {col_type}"))
                conn.commit()


def _migrate_experiment_metrics_table():
    """实验指标表迁移：将 extra_data 列从 VARCHAR(1024) 扩容为 TEXT

    SQLite 不支持 ALTER COLUMN，需通过重建表来修改列类型。
    检测方式：通过 PRAGMA table_info 检查 type 是否为 TEXT 类（含 TEXT/CLOB）。
    """
    from sqlalchemy import text, inspect
    inspector = inspect(engine)

    if "experiment_metrics" not in inspector.get_table_names():
        return

    cols = inspector.get_columns("experiment_metrics")
    extra_col = next((c for c in cols if c["name"] == "extra_data"), None)
    if not extra_col:
        return

    col_type = str(extra_col["type"]).upper()
    if "TEXT" in col_type or "CLOB" in col_type:
        return

    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=OFF"))
        try:
            conn.execute(text("""
                CREATE TABLE experiment_metrics_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id VARCHAR(64) NOT NULL,
                    step INTEGER NOT NULL DEFAULT 0,
                    epoch INTEGER DEFAULT 0,
                    loss FLOAT DEFAULT 0.0,
                    accuracy FLOAT DEFAULT 0.0,
                    val_loss FLOAT DEFAULT 0.0,
                    val_accuracy FLOAT DEFAULT 0.0,
                    learning_rate FLOAT DEFAULT 0.0,
                    batch_size INTEGER DEFAULT 0,
                    metric_type VARCHAR(32) NOT NULL DEFAULT 'training',
                    extra_data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                INSERT INTO experiment_metrics_new
                    (id, experiment_id, step, epoch, loss, accuracy, val_loss, val_accuracy,
                     learning_rate, batch_size, metric_type, extra_data, created_at)
                SELECT id, experiment_id, step, epoch, loss, accuracy, val_loss, val_accuracy,
                       learning_rate, batch_size, metric_type, extra_data, created_at
                FROM experiment_metrics
            """))
            conn.execute(text("DROP TABLE experiment_metrics"))
            conn.execute(text("ALTER TABLE experiment_metrics_new RENAME TO experiment_metrics"))
            conn.execute(text("CREATE INDEX ix_experiment_metrics_experiment_id ON experiment_metrics(experiment_id)"))
            conn.commit()
        except Exception as e:
            conn.rollback()
            import logging
            logging.getLogger(__name__).error(f"experiment_metrics 表迁移失败: {e}")
        finally:
            conn.execute(text("PRAGMA foreign_keys=ON"))


def get_db():
    """获取数据库会话的依赖"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
