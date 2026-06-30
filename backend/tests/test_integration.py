"""
端到端集成测试
覆盖: 数据加载 → 模型构建 → 训练循环 → checkpoint → 恢复 → 评估
不依赖数据库，纯内存操作验证核心流程
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from app.ml.model_builder import build_model, ConfigurableCNN, ConfigurableMLP
from app.ml.losses import create_loss_function
from app.ml.attention import create_attention
from app.ml.training_utils import (
    CheckpointManager, EarlyStopping, create_scheduler, AMPManager,
)
from app.ml.augmentation import (
    get_augmentation_transform, apply_augmentation_batch,
    cutmix_data, mixup_data,
)
from app.ml.metrics import compute_roc_curve, compute_pr_curve


class TestEndToEndPipeline:
    """端到端训练流程测试"""

    def test_full_cnn_training_pipeline(self):
        """CNN 完整训练流程：数据 → 模型 → 训练 → checkpoint → 恢复"""
        torch.manual_seed(42)

        # 1. 准备数据
        X = torch.randn(500, 1, 28, 28)
        y = torch.randint(0, 5, (500,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 2. 构建模型（带注意力）
        model = build_model("image_folder", "1x28x28", 5, {
            "channels": [16, 32],
            "attention": "se",
            "use_bn": True,
        })
        assert isinstance(model, ConfigurableCNN)

        # 3. 损失函数和优化器
        criterion = create_loss_function("cross_entropy")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 4. 训练 3 个 epoch
        model.train()
        losses = []
        for epoch in range(3):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(train_loader))

        # 验证 loss 下降
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

        # 5. 保存 checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(tmpdir)
            ckpt.save(model, optimizer, epoch=3, best_val_acc=0.8, best_epoch=3)

            # 6. 修改参数后恢复
            for p in model.parameters():
                p.data.zero_()
            restored = ckpt.load(model, optimizer)
            assert restored is not None
            assert restored["epoch"] == 3
            assert not torch.allclose(
                next(model.parameters()),
                torch.zeros_like(next(model.parameters()))
            )

            # 7. 对恢复后的模型评估
            model.eval()
            with torch.no_grad():
                output = model(X[:100])
                acc = (output.argmax(dim=1) == y[:100]).float().mean().item()
            assert acc > 0.0

    def test_full_mlp_training_pipeline(self):
        """MLP 完整训练流程"""
        torch.manual_seed(42)

        X = torch.randn(400, 64)
        y = torch.randint(0, 3, (400,))
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = build_model("csv", "64", 3)
        assert isinstance(model, ConfigurableMLP)

        criterion = create_loss_function("focal")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(3):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(X)
            preds = output.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        assert acc > 0.2  # 基本学习能力

    def test_training_with_augmentation(self):
        """训练 + 数据增强"""
        torch.manual_seed(42)

        X = np.random.rand(200, 1, 28, 28).astype(np.float32)
        y = np.random.randint(0, 5, 200).astype(np.int64)

        # 配置增强
        transform = get_augmentation_transform({
            "horizontal_flip": True,
            "random_erasing": True,
        }, is_train=True)

        model = ConfigurableCNN(in_channels=1, num_classes=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(2):
            # 批量增强
            X_aug = apply_augmentation_batch(X, transform)
            X_tensor = torch.from_numpy(X_aug).float()
            y_tensor = torch.from_numpy(y).long()

            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        assert loss.item() > 0

    def test_training_with_cutmix(self):
        """训练 + CutMix"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = np.random.rand(32, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 5, 32).astype(np.int64)

        # CutMix
        mixed_images, mixed_labels = cutmix_data(X, labels, alpha=1.0, num_classes=5)
        X_tensor = torch.from_numpy(mixed_images).float()
        y_tensor = torch.from_numpy(mixed_labels).float()

        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_training_with_mixup(self):
        """训练 + MixUp"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = np.random.rand(32, 1, 28, 28).astype(np.float32)
        labels = np.random.randint(0, 5, 32).astype(np.int64)

        mixed_images, mixed_labels = mixup_data(X, labels, alpha=1.0, num_classes=5)
        X_tensor = torch.from_numpy(mixed_images).float()
        y_tensor = torch.from_numpy(mixed_labels).float()

        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_full_training_with_early_stopping(self):
        """完整训练 + 早停"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        es = EarlyStopping(patience=3, mode="max")

        X = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 3, (100,))

        stopped_epoch = 0
        for epoch in range(1, 30):
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
        assert stopped_epoch < 30

    def test_full_training_with_scheduler(self):
        """完整训练 + 学习率调度"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler, _ = create_scheduler(optimizer, "cosine", total_epochs=10)

        initial_lr = optimizer.param_groups[0]["lr"]
        X = torch.randn(50, 1, 28, 28)
        y = torch.randint(0, 3, (50,))

        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_checkpoint_resume_integration(self):
        """Checkpoint 保存 + 恢复 + 继续训练"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(50, 1, 28, 28)
        y = torch.randint(0, 3, (50,))

        # 训练 2 个 epoch
        model.train()
        for epoch in range(2):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(tmpdir)
            ckpt.save(model, optimizer, epoch=2, best_val_acc=0.5, best_epoch=2)

            # 创建新模型
            model2 = ConfigurableCNN(in_channels=1, num_classes=3)
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

            restored = ckpt.load(model2, optimizer2)
            assert restored["epoch"] == 2

            # 继续训练 1 个 epoch
            model2.train()
            optimizer2.zero_grad()
            output = model2(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer2.step()

            # 保存 epoch 3
            ckpt.save(model2, optimizer2, epoch=3, best_val_acc=0.6, best_epoch=3)

            # 恢复到 epoch 3
            model3 = ConfigurableCNN(in_channels=1, num_classes=3)
            restored3 = ckpt.load(model3)
            assert restored3["epoch"] == 3

    def test_metrics_computation(self):
        """训练后评估指标计算"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        model.eval()

        X = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 3, (100,))

        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)

        y_true = y.tolist()
        y_scores = probs.tolist()

        # ROC
        roc = compute_roc_curve(y_true, y_scores, 3)
        assert roc["num_classes"] == 3
        assert 0.0 <= roc["macro_auc"] <= 1.0

        # PR
        pr = compute_pr_curve(y_true, y_scores, 3)
        assert pr["num_classes"] == 3
        assert 0.0 <= pr["macro_ap"] <= 1.0

    def test_multi_loss_training(self):
        """多损失函数组合训练"""
        torch.manual_seed(42)

        from app.ml.losses import CombinedLoss, FocalLoss, DiceLoss

        model = ConfigurableCNN(in_channels=1, num_classes=5)
        criterion = CombinedLoss(losses=[
            ("ce", nn.CrossEntropyLoss(), 0.7),
            ("focal", FocalLoss(alpha=0.25), 0.3),
        ])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 5, (64,))

        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        # 验证损失详情
        details = criterion.get_loss_details()
        assert len(details) == 2

    def test_attention_integration(self):
        """注意力机制在训练中集成"""
        torch.manual_seed(42)

        for attn_type in ["se", "cbam", "self_attention", "gct", "coord"]:
            model = ConfigurableCNN(in_channels=1, num_classes=3, attention=attn_type)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            X = torch.randn(32, 1, 28, 28)
            y = torch.randint(0, 3, (32,))

            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            assert loss.item() > 0

    def test_loss_function_training(self):
        """不同损失函数在训练中集成"""
        torch.manual_seed(42)

        for loss_name in ["cross_entropy", "focal", "label_smoothing", "dice"]:
            model = ConfigurableCNN(in_channels=1, num_classes=5)
            criterion = create_loss_function(loss_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            X = torch.randn(32, 1, 28, 28)
            y = torch.randint(0, 5, (32,))

            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            assert loss.item() > 0, f"Loss {loss_name} failed"

    def test_amp_integration(self):
        """混合精度训练集成"""
        torch.manual_seed(42)

        model = ConfigurableCNN(in_channels=1, num_classes=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        amp = AMPManager(enabled=True)

        X = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 3, (16,))

        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            with amp.autocast_context():
                output = model(X)
                loss = criterion(output, y)
            amp.backward(loss)
            amp.optimizer_step(optimizer)

        assert loss.item() > 0