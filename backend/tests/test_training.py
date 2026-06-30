"""
训练流程集成测试
覆盖: 完整训练循环、早停触发、checkpoint 续训、学习率调度、混合精度
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from app.ml.training_utils import EarlyStopping, CheckpointManager, AMPManager
from app.ml.model_builder import ConfigurableCNN, ConfigurableMLP


class TestTrainingLoop:
    """测试训练循环核心逻辑"""

    @pytest.fixture
    def simple_model(self):
        """简单的 CNN 模型"""
        return ConfigurableCNN(in_channels=1, num_classes=5)

    @pytest.fixture
    def training_data(self):
        """生成训练数据"""
        torch.manual_seed(42)
        X = torch.randn(200, 1, 28, 28)
        y = torch.randint(0, 5, (200,))
        return X, y

    def test_one_epoch_trains(self, simple_model, training_data):
        """一个 epoch 训练后参数更新"""
        X, y = training_data
        model = simple_model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 保存初始参数
        initial_params = [p.clone() for p in model.parameters()]

        # 训练一个 epoch
        model.train()
        for i in range(0, len(X), 32):
            batch_X = X[i:i+32]
            batch_y = y[i:i+32]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        # 验证参数已更新
        for p_init, p_new in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_init, p_new)

    def test_model_overfits_small_data(self, simple_model, training_data):
        """小数据集上模型能过拟合"""
        X, y = training_data
        # 只取 10 个样本
        X_small = X[:10]
        y_small = y[:10]

        model = simple_model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 训练多个 epoch
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = model(X_small)
            loss = criterion(output, y_small)
            loss.backward()
            optimizer.step()

        # 验证模型能完美预测训练数据
        model.eval()
        with torch.no_grad():
            output = model(X_small)
            preds = output.argmax(dim=1)
            accuracy = (preds == y_small).float().mean().item()

        assert accuracy > 0.9, f"Expected accuracy > 0.9, got {accuracy}"

    def test_loss_decreases(self, simple_model, training_data):
        """训练过程中 loss 趋势下降"""
        X, y = training_data
        model = simple_model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        losses = []
        model.train()
        for epoch in range(20):
            epoch_losses = []
            for i in range(0, len(X), 32):
                batch_X = X[i:i+32]
                batch_y = y[i:i+32]
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))

        # 前几个 epoch 的平均 loss 应大于后几个 epoch
        early_avg = np.mean(losses[:5])
        late_avg = np.mean(losses[-5:])
        assert late_avg < early_avg, f"Loss did not decrease: early={early_avg:.4f}, late={late_avg:.4f}"


class TestEarlyStoppingIntegration:
    """测试早停集成"""

    def test_early_stopping_stops_training(self):
        """早停在实际训练中生效"""
        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        X = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 3, (100,))

        es = EarlyStopping(patience=3, mode="max", verbose=False)
        stopped_epoch = 0

        for epoch in range(1, 50):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X)
                val_acc = (val_output.argmax(dim=1) == y).float().mean().item()

            if es(val_acc, epoch):
                stopped_epoch = epoch
                break

        assert stopped_epoch > 0
        assert stopped_epoch < 50  # 应该在 patience 内停止


class TestCheckpointIntegration:
    """测试 checkpoint 集成"""

    def test_checkpoint_resume_preserves_state(self):
        """checkpoint 恢复后状态一致"""
        import tempfile
        import os

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            mgr.save(model, optimizer, epoch=3, best_val_acc=0.75, best_epoch=2)

            # 修改参数
            for p in model.parameters():
                p.data.zero_()

            # 恢复
            ckpt = mgr.load(model, optimizer)
            assert ckpt is not None
            assert ckpt["epoch"] == 3
            # 参数不应全零
            assert not torch.allclose(
                next(model.parameters()),
                torch.zeros_like(next(model.parameters()))
            )


class TestLRIntegration:
    """测试学习率调度集成"""

    def test_cosine_scheduler_in_loop(self):
        """CosineAnnealingLR 在训练循环中工作"""
        model = ConfigurableCNN(in_channels=1, num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        initial_lr = optimizer.param_groups[0]["lr"]
        # 模拟一个训练step：forward + backward + optimizer.step + scheduler.step
        dummy_input = torch.randn(1, 1, 28, 28)
        dummy_target = torch.randint(0, 3, (1,))
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(dummy_input), dummy_target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_plateau_scheduler_in_loop(self):
        """ReduceLROnPlateau 在训练循环中工作"""
        model = ConfigurableCNN(in_channels=1, num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        initial_lr = optimizer.param_groups[0]["lr"]
        # 模拟验证 loss 不改善，patience=2 需要 3 次 step 触发
        for _ in range(4):
            scheduler.step(0.5)

        # patience=2, 所以第3次应该触发，第4次确认
        assert optimizer.param_groups[0]["lr"] <= initial_lr * 0.5


class TestAMPIntegration:
    """测试 AMP 集成"""

    def test_amp_training_loop(self):
        """AMP 在训练循环中工作"""
        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        amp = AMPManager(enabled=True)

        X = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 3, (4,))

        model.train()
        optimizer.zero_grad()

        with amp.autocast_context():
            output = model(X)
            loss = criterion(output, y)

        amp.backward(loss)
        amp.optimizer_step(optimizer)

        # 训练不应崩溃
        assert loss.item() > 0


class TestMLPTraining:
    """测试 MLP 训练"""

    def test_mlp_training_loop(self):
        """MLP 完整训练循环"""
        model = ConfigurableMLP(in_features=64, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        X = torch.randn(100, 64)
        y = torch.randint(0, 3, (100,))

        model.train()
        for epoch in range(5):
            for i in range(0, len(X), 16):
                batch_X = X[i:i+16]
                batch_y = y[i:i+16]
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(X)
            acc = (output.argmax(dim=1) == y).float().mean().item()
        assert acc > 0.3  # 基本的学习能力