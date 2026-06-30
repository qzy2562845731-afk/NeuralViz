"""
训练工具模块单元测试
覆盖: CheckpointManager, EarlyStopping, create_scheduler, AMPManager
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from app.ml.training_utils import (
    CheckpointManager, EarlyStopping, create_scheduler, AMPManager,
)


class TestCheckpointManager:
    """测试断点续训管理器"""

    @pytest.fixture
    def model_and_optimizer(self):
        """创建测试模型和优化器"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_save_and_load(self, model_and_optimizer, temp_dir):
        """保存和加载 checkpoint"""
        model, optimizer = model_and_optimizer
        mgr = CheckpointManager(temp_dir)

        # 保存
        mgr.save(model, optimizer, epoch=5, best_val_acc=0.85, best_epoch=3)
        assert mgr.exists()

        # 修改模型参数
        for p in model.parameters():
            p.data.zero_()

        # 加载
        ckpt = mgr.load(model, optimizer)
        assert ckpt is not None
        assert ckpt["epoch"] == 5
        assert ckpt["best_val_acc"] == 0.85
        assert ckpt["best_epoch"] == 3

        # 验证参数已恢复
        assert not torch.allclose(
            next(model.parameters()),
            torch.zeros_like(next(model.parameters()))
        )

    def test_load_nonexistent(self, temp_dir):
        """加载不存在的 checkpoint"""
        mgr = CheckpointManager(temp_dir)
        model = nn.Linear(10, 5)
        ckpt = mgr.load(model)
        assert ckpt is None

    def test_exists(self, model_and_optimizer, temp_dir):
        """检查 checkpoint 是否存在"""
        model, optimizer = model_and_optimizer
        mgr = CheckpointManager(temp_dir)
        assert not mgr.exists()
        mgr.save(model, optimizer, epoch=1, best_val_acc=0.5, best_epoch=1)
        assert mgr.exists()

    def test_remove(self, model_and_optimizer, temp_dir):
        """删除 checkpoint"""
        model, optimizer = model_and_optimizer
        mgr = CheckpointManager(temp_dir)
        mgr.save(model, optimizer, epoch=1, best_val_acc=0.5, best_epoch=1)
        assert mgr.exists()
        mgr.remove()
        assert not mgr.exists()

    def test_save_with_scheduler(self, model_and_optimizer, temp_dir):
        """保存带 scheduler 的 checkpoint"""
        model, optimizer = model_and_optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        mgr = CheckpointManager(temp_dir)

        mgr.save(model, optimizer, epoch=5, best_val_acc=0.9, best_epoch=5, scheduler=scheduler)
        assert mgr.exists()

        # 模拟一个训练step后修改 scheduler 状态
        dummy_input = torch.randn(1, 10)
        dummy_target = torch.randn(1, 5)  # 模型输出维度为5
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(dummy_input), dummy_target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 加载恢复
        ckpt = mgr.load(model, optimizer, scheduler=scheduler)
        assert ckpt is not None

    def test_save_with_extra(self, model_and_optimizer, temp_dir):
        """保存带额外元数据的 checkpoint"""
        model, optimizer = model_and_optimizer
        mgr = CheckpointManager(temp_dir)

        extra = {"learning_rate": 0.001, "scheduler_type": "cosine"}
        mgr.save(model, optimizer, epoch=3, best_val_acc=0.8, best_epoch=2, extra=extra)
        assert mgr.exists()

        ckpt = mgr.load(model, optimizer)
        assert ckpt is not None
        assert ckpt.get("extra", {}).get("learning_rate") == 0.001


class TestEarlyStopping:
    """测试早停机制"""

    def test_no_stop_initially(self):
        """初始状态不应停止"""
        es = EarlyStopping(patience=5, mode="max")
        assert not es(0.8, epoch=1)

    def test_improvement_resets_counter(self):
        """改善时重置计数器"""
        es = EarlyStopping(patience=3, mode="max")
        es(0.8, epoch=1)
        es(0.75, epoch=2)  # 下降 1
        es(0.76, epoch=2)  # 下降 2
        assert es.counter == 2
        es(0.85, epoch=3)  # 改善
        assert es.counter == 0

    def test_stop_after_patience(self):
        """patience 后停止"""
        es = EarlyStopping(patience=3, mode="max")
        es(0.8, epoch=1)
        for i in range(2, 5):
            result = es(0.7, epoch=i)
        assert result is True
        assert es.early_stop

    def test_mode_min(self):
        """监控 loss 下降模式"""
        es = EarlyStopping(patience=3, mode="min")
        es(0.5, epoch=1)
        assert not es(0.55, epoch=2)  # 上升，视为恶化
        assert es.counter == 1

    def test_min_delta(self):
        """min_delta 阈值"""
        es = EarlyStopping(patience=3, mode="max", min_delta=0.01)
        es(0.8, epoch=1)
        es(0.805, epoch=2)  # 改善 0.005 < min_delta
        assert es.counter == 1
        es(0.82, epoch=3)  # 改善 0.02 > min_delta
        assert es.counter == 0

    def test_reset(self):
        """重置早停状态"""
        es = EarlyStopping(patience=3, mode="max")
        es(0.8, epoch=1)
        es(0.7, epoch=2)
        es.reset()
        assert es.counter == 0
        assert es.best_score is None
        assert not es.early_stop

    def test_best_epoch_tracking(self):
        """追踪最佳 epoch"""
        es = EarlyStopping(patience=5, mode="max")
        es(0.7, epoch=1)
        es(0.8, epoch=3)
        es(0.75, epoch=5)
        assert es.best_epoch == 3


class TestCreateScheduler:
    """测试学习率调度器工厂"""

    def test_cosine_scheduler(self):
        """CosineAnnealingLR"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(optimizer, "cosine", total_epochs=20)
        assert scheduler is not None
        assert s_type == "cosine_annealing"
        # 模拟训练step后再调用scheduler.step()
        dummy_input = torch.randn(1, 10)
        dummy_target = torch.randn(1, 5)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(dummy_input), dummy_target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] < 0.1

    def test_plateau_scheduler(self):
        """ReduceLROnPlateau"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(
            optimizer, "plateau",
            scheduler_params={"mode": "max", "factor": 0.5, "patience": 2}
        )
        assert scheduler is not None
        assert s_type == "reduce_on_plateau"
        # Plateau scheduler 需要 metric 参数
        scheduler.step(0.8)

    def test_step_scheduler(self):
        """StepLR"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(
            optimizer, "step",
            scheduler_params={"step_size": 5, "gamma": 0.5}
        )
        assert scheduler is not None
        assert s_type == "step"

    def test_exponential_scheduler(self):
        """ExponentialLR"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(
            optimizer, "exponential",
            scheduler_params={"gamma": 0.95}
        )
        assert scheduler is not None
        assert s_type == "exponential"

    def test_none_scheduler(self):
        """无调度器"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(optimizer, "none")
        assert scheduler is None
        assert s_type == "none"

    def test_invalid_scheduler_fallback(self):
        """无效调度器类型回退"""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, s_type = create_scheduler(optimizer, "invalid_type")
        assert scheduler is None
        assert s_type == "none"


class TestAMPManager:
    """测试混合精度管理器"""

    def test_amp_disabled_on_cpu(self):
        """CPU 环境下 AMP 自动禁用"""
        amp = AMPManager(enabled=True)
        # 在 CPU 上 AMP 应该被禁用
        assert not amp.enabled
        assert amp.scaler is None

    def test_amp_disabled_explicitly(self):
        """显式禁用 AMP"""
        amp = AMPManager(enabled=False)
        assert not amp.enabled
        assert amp.scaler is None

    def test_autocast_context_cpu(self):
        """CPU 下 autocast 返回 nullcontext"""
        amp = AMPManager(enabled=True)
        ctx = amp.autocast_context()
        assert ctx is not None

    def test_backward_cpu(self):
        """CPU 下 backward 正常工作"""
        amp = AMPManager(enabled=True)
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        # 不应抛出异常
        amp.backward(loss)

    def test_optimizer_step_cpu(self):
        """CPU 下 optimizer_step 正常工作"""
        amp = AMPManager(enabled=True)
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        amp.optimizer_step(optimizer)

    def test_state_dict_none_on_cpu(self):
        """CPU 下 state_dict 返回 None"""
        amp = AMPManager(enabled=True)
        assert amp.state_dict() is None

    def test_load_state_dict_handles_none(self):
        """load_state_dict 处理 None"""
        amp = AMPManager(enabled=True)
        # 不应抛出异常
        amp.load_state_dict(None)