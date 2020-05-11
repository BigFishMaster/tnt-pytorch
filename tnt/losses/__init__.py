from torch.nn import CrossEntropyLoss
from torch.nn import SoftMarginLoss
from .losses import MultiLabelLoss
from .losses import WeightLabelLoss
from .losses import RelativeLabelLoss, RelativeLabelLossV2
from .class_balanced_loss import ClassBalancedLoss
from .face_loss import ArcFace, CosFace