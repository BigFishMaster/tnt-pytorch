import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricCELoss(nn.Module):

    def __init__(self, feature_size, num_classes):
        """
        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(MetricCELoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        output = F.linear(F.relu(feature), self.weight)
        return output

    def forward(self, feature, label):
        output = self.output(feature)
        loss = self.ce(output, label)
        return loss, output
