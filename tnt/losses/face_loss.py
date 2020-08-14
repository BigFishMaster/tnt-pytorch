# Definition of ArcFace loss and CosFace loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):

    def __init__(self, feature_size, num_classes, s=30.0, m=0.50):
        """ArcFace formula:
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether
        (m + theta) go out of [0, Pi]

        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
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
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
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

    def __init__(self, feature_size, num_classes, s=30.0, m=0.40):
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
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
        if len(label) == len(feature) * 16:
            B, C = feature.shape[:2]
            feature = feature.reshape(B*16, C//16)
        cosine = self.output(feature)
        phi = cosine - self.m
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        loss = self.ce(output * self.s, label)
        return loss, cosine


class MultipleCosFaceLoss(nn.Module):

    def __init__(self, feature_size, num_classes, s=30.0, m=0.40):
        """
        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(MultipleCosFaceLoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(1, 16, self.in_features, self.out_features))
        nn.init.xavier_uniform_(self.weight)

        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        # feature:      B x 16 x norm(C) x 1
        # weight:       1 x 16 x norm(C) x D
        # cosine:       B x 16 x C x D
        cosine = F.normalize(feature, dim=2) * F.normalize(self.weight, dim=2)
        # cosine: B x 16 x D
        cosine = cosine.sum(dim=2)
        return cosine

    def forward(self, feature, target):
        # f: B x C*16
        # l: B*16
        B, C = feature.shape[:2]
        feature = feature.reshape(B, 16, C//16, 1)
        # B x 16 x D
        cosines = self.output(feature)
        #
        labels = target.reshape(B, 16)
        total_loss = 0
        for i in range(16):
            cosine = cosines[:, i, :]
            label = labels[:, i]
            phi = cosine - self.m
            output = cosine * 1.0  # make backward works
            batch_size = len(output)
            output[range(batch_size), label] = phi[range(batch_size), label]
            loss = self.ce(output * self.s, label)
            total_loss += loss
        total_loss = total_loss / 16
        out_cosine = cosines.reshape(B*16, -1)
        return total_loss.unsqueeze(0), out_cosine


class CosFaceLossWithNeg(nn.Module):

    def __init__(self, feature_size, num_classes, s=30.0, m=0.40):
        """
        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(CosFaceLossWithNeg, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        cosine = F.linear(F.normalize(feature), F.normalize(self.weight))
        return cosine

    def forward(self, feature, label):
        cosine = self.output(feature)

        neg_index = label >= self.out_features
        pos_index = label < self.out_features
        neg_num = torch.sum(neg_index)
        pos_num = torch.sum(pos_index)
        loss = 0
        if neg_num > 0:
            neg_cosine = cosine[neg_index]
            neg_max_score, neg_max_index = F.softmax(neg_cosine, dim=1).max(dim=1)
            loss_neg = -torch.log(1 - neg_max_score).mean()
            loss += loss_neg

        if pos_num > 0:
            pos_cosine = cosine[pos_index]
            pos_label = label[pos_index]
            phi = pos_cosine - self.m
            output = pos_cosine * 1.0  # make backward works
            batch_size = len(output)
            output[range(batch_size), pos_label] = phi[range(batch_size), pos_label]
            loss_pos = self.ce(output * self.s, pos_label)
            loss += loss_pos

        return loss, cosine


if __name__ == "__main__":
    feature = torch.rand((10, 2048))
    label = torch.randint(0, 100, size=(160,))

    loss_func = MultipleCosFaceLoss(128, 100)
    output = loss_func(feature, label)
    print("ok.")

