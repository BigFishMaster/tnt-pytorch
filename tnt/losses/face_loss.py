# Definition of ArcFace loss and CosFace loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ The implementation of Arcface paper: https://arxiv.org/abs/1801.07698

    In ArcFace formulation:

    :math:`cos(m + \\theta) = cos(m)cos(\\theta) - sin(m)sin(\\theta)`

    Note that: :math:`0 <= m + \\theta <= \pi`

    So if :math:`(m + \\theta) >= \pi`, then :math:`\\theta >= \pi - m`. In :math:`[0, \pi]`,
    we have: :math:`cos(\\theta) < cos(\pi - m)`

    So we can use :math:`cos(\pi - m)` as threshold to check whether :math:`(m + \\theta)` go out of :math:`[0, \pi]`.

    Args:
        feature_size (int): usually 128, 256, 512 ...
        num_classes (int): num of faces when training
        s (float): loss scale. Default: ``30.0``. See Normface(https://arxiv.org/abs/1704.06369).
        m (float): loss margin. Default: ``0.5``. See related papers for explanation:

            * ``Arcface``: https://arxiv.org/abs/1801.07698
            * ``Cosface``: https://arxiv.org/abs/1801.09414
            * ``Sphereface``: https://arxiv.org/abs/1704.08063
    """
    def __init__(self, feature_size, num_classes, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        """ Output the logit.

        Args:
            feature (:obj:`torch.FloatTensor`): face features from previous layers.

        Returns:
            :obj:`torch.FloatTensor`: cosine logit
        """
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
        """ Calculate Arcface Loss.

        Args:
            feature (:obj:`torch.FloatTensor`): face features from previous layers.
            label (:obj:`torch.LongTensor`): face labels.

        Returns:
            (:obj:`torch.FloatTensor`, :obj:`torch.FloatTensor`):

            * loss: Arcface loss over the batch.
            * logit: cosine logit output.
        """
        cosine = self.output(feature)
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss = self.ce(output * self.s, label)
        return loss, cosine


class CosFaceLoss(nn.Module):
    """ The implementation of Cosface paper: https://arxiv.org/abs/1801.09414

    Args:
        feature_size (int): usually 128, 256, 512 ...
        num_classes (int): num of faces when training
        s (float): loss scale. Default: ``30.0``. See Normface(https://arxiv.org/abs/1704.06369).
        m (float): loss margin. Default: ``0.4``. See related papers for explanation:

            * ``Arcface``: https://arxiv.org/abs/1801.07698
            * ``Cosface``: https://arxiv.org/abs/1801.09414
            * ``Sphereface``: https://arxiv.org/abs/1704.08063
    """
    def __init__(self, feature_size, num_classes, s=30.0, m=0.40):
        super(CosFaceLoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        """ Output the logit.

        Args:
            feature (:obj:`torch.FloatTensor`): face features from previous layers.

        Returns:
            :obj:`torch.FloatTensor`: cosine logit
        """
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
        """ Calculate Cosface Loss.

        Args:
            feature (:obj:`torch.FloatTensor`): face features from previous layers.
            label (:obj:`torch.LongTensor`): face labels.

        Returns:
            (:obj:`torch.FloatTensor`, :obj:`torch.FloatTensor`):

            * loss: Cosface loss over the batch.
            * logit: cosine logit output.
        """
        cosine = self.output(feature)
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss = self.ce(output * self.s, label)
        return loss, cosine
