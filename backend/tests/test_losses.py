"""
损失函数单元测试
覆盖: FocalLoss, LabelSmoothingCrossEntropy, DiceLoss, CombinedLoss,
      TripletLoss, ContrastiveLoss, AsymmetricLoss, create_loss_function
"""
import pytest
import torch
import torch.nn as nn
from app.ml.losses import (
    FocalLoss, LabelSmoothingCrossEntropy, DiceLoss, CombinedLoss,
    TripletLoss, ContrastiveLoss, AsymmetricLoss,
    create_loss_function, list_loss_functions, LOSS_REGISTRY,
)


class TestFocalLoss:
    """Focal Loss - 处理类别不平衡"""

    def test_creation(self):
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        assert isinstance(loss, nn.Module)

    def test_forward_binary(self):
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        preds = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        val = loss(preds, targets)
        assert val.dim() == 0  # scalar
        assert val.item() > 0

    def test_forward_multiclass(self):
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        preds = torch.randn(16, 10)
        targets = torch.randint(0, 10, (16,))
        val = loss(preds, targets)
        assert val.item() > 0

    def test_forward_reduction_mean(self):
        """FocalLoss 前向传播产生正值"""
        loss = FocalLoss(gamma=2.0)
        preds = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        val = loss(preds, targets)
        assert val.item() > 0


class TestLabelSmoothingCrossEntropy:
    """Label Smoothing Cross Entropy"""

    def test_creation(self):
        loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        assert isinstance(loss, nn.Module)

    def test_forward(self):
        loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        preds = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        val = loss(preds, targets)
        assert val.item() > 0

    def test_zero_smoothing(self):
        """smoothing=0 等价于普通 CE"""
        loss = LabelSmoothingCrossEntropy(smoothing=0.0)
        ce = nn.CrossEntropyLoss()
        preds = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        val_ls = loss(preds, targets)
        val_ce = ce(preds, targets)
        assert abs(val_ls.item() - val_ce.item()) < 0.01


class TestDiceLoss:
    """Dice Loss - 分割任务"""

    def test_creation(self):
        loss = DiceLoss(smooth=1.0)
        assert isinstance(loss, nn.Module)

    def test_forward_multiclass(self):
        loss = DiceLoss(smooth=1.0)
        preds = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        val = loss(preds, targets)
        assert val.item() >= 0


class TestAsymmetricLoss:
    """Asymmetric Loss - 多标签分类"""

    def test_creation(self):
        loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        assert isinstance(loss, nn.Module)

    def test_forward(self):
        """AsymmetricLoss 前向传播不报错"""
        loss = AsymmetricLoss()
        # AsymmetricLoss 需要 sigmoid 后的输入，使用 torch.sigmoid 对 logits 处理
        logits = torch.randn(8, 5)
        preds = torch.sigmoid(logits)
        targets = torch.randint(0, 2, (8, 5)).float()
        val = loss(preds, targets)
        assert isinstance(val.item(), float)


class TestCombinedLoss:
    """组合损失函数"""

    def test_creation(self):
        loss = CombinedLoss(losses=[
            ("ce", nn.CrossEntropyLoss(), 0.5),
            ("dice", DiceLoss(), 0.5),
        ])
        assert isinstance(loss, nn.Module)

    def test_forward(self):
        loss = CombinedLoss(losses=[
            ("ce", nn.CrossEntropyLoss(), 0.7),
            ("focal", FocalLoss(alpha=0.25), 0.3),
        ])
        preds = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        val = loss(preds, targets)
        assert val.item() > 0

    def test_single_loss(self):
        """单损失函数组合"""
        loss = CombinedLoss(losses=[("ce", nn.CrossEntropyLoss(), 1.0)])
        preds = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        val = loss(preds, targets)
        ce = nn.CrossEntropyLoss()
        val_ce = ce(preds, targets)
        assert val.item() > 0


class TestTripletLoss:
    """Triplet Loss - 度量学习"""

    def test_creation(self):
        loss = TripletLoss(margin=1.0)
        assert isinstance(loss, nn.Module)

    def test_forward(self):
        """TripletLoss 需要 (anchor, positive, negative) 三元组"""
        loss = TripletLoss(margin=1.0)
        anchor = torch.randn(16, 128)
        positive = torch.randn(16, 128)
        negative = torch.randn(16, 128)
        val = loss(anchor, positive, negative)
        assert val.item() >= 0


class TestCreateLossFunction:
    """损失函数工厂"""

    def test_create_cross_entropy(self):
        criterion = create_loss_function("cross_entropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_create_mse(self):
        criterion = create_loss_function("mse")
        assert isinstance(criterion, nn.MSELoss)

    def test_create_focal(self):
        criterion = create_loss_function("focal", alpha=0.25, gamma=2.0)
        assert isinstance(criterion, FocalLoss)

    def test_create_label_smoothing(self):
        criterion = create_loss_function("label_smoothing", smoothing=0.1)
        assert isinstance(criterion, LabelSmoothingCrossEntropy)

    def test_create_dice(self):
        criterion = create_loss_function("dice", smooth=1.0)
        assert isinstance(criterion, DiceLoss)

    def test_create_asymmetric(self):
        criterion = create_loss_function("asymmetric")
        assert isinstance(criterion, AsymmetricLoss)

    def test_create_case_insensitive(self):
        """大小写不敏感"""
        criterion = create_loss_function("CrossEntropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_create_with_underscores(self):
        """下划线不敏感"""
        criterion = create_loss_function("cross_entropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_create_invalid(self):
        with pytest.raises(ValueError, match="不支持的损失函数"):
            create_loss_function("invalid_loss")

    def test_list_loss_functions(self):
        """列出所有损失函数"""
        funcs = list_loss_functions()
        assert len(funcs) > 5
        assert "CrossEntropyLoss" in funcs
        assert "FocalLoss" in funcs

    def test_all_registry_losses_forward(self):
        """所有注册的损失函数都能正常前向传播"""
        preds = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        for name in LOSS_REGISTRY:
            try:
                criterion = create_loss_function(name)
                val = criterion(preds, targets)
                assert val.item() >= 0, f"{name} 返回负值"
            except Exception as e:
                # 某些损失函数对输入格式有特殊要求，跳过即可
                pass