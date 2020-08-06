import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFaceLoss(nn.Module):

    def __init__(self, feature_size, num_classes, s=30.0, m=0.40, use_gpu=True):
        """
        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(CosFaceLoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        if use_gpu:
            self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).cuda())
        else:
            self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
        cosine = self.output(feature)
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss = self.ce(output * self.s, label)
        return loss, cosine

