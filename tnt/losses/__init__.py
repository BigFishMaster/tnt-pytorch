from torch.nn import CrossEntropyLoss
from torch.nn import SoftMarginLoss
from .losses import PseudoLabelLoss
from .losses import MultiLabelLoss
from .losses import WeightLabelLoss
from .losses import RelativeLabelLoss, RelativeLabelLossV2
from .class_balanced_loss import ClassBalancedLoss
from .face_loss import ArcFaceLoss, CosFaceLoss
from .metric_loss import MetricCELoss, HCLoss

__all__ = [
    "CrossEntropyLoss", "SoftMarginLoss", "PseudoLabelLoss", "MultiLabelLoss",
    "WeightLabelLoss", "RelativeLabelLossV2", "RelativeLabelLoss",
    "ClassBalancedLoss", "ArcFaceLoss", "CosFaceLoss", "MetricCELoss",
    "HCLoss"
]

parameterized_losses = ["ArcFaceLoss", "CosFaceLoss", "MetricCELoss"]
initialized_losses = ["ClassBalancedLoss"]
