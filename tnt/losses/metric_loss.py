import torch
import torch.nn as nn
import torch.nn.functional as F
from tnt.utils.logging import logger, init_logger


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


class MultiCELoss(nn.Module):

    def __init__(self, feature_size, num_classes):
        """
        Args:
            feature_size: usually 128, 256, 512 ...
            num_classes: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        """
        super(MultiCELoss, self).__init__()
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


class HCLoss(nn.Module):

    def __init__(self, each_class, beta=10000.0, pos_nn=1.0, sample_type="ratio", margin=1.0):
        super(HCLoss, self).__init__()
        self.each_class = each_class
        self.beta = beta
        self.pos_num = int(pos_nn * (each_class-1))
        self.sample_type = sample_type
        self.margin = margin
        self.max_gap = 20
        logger.info("Initialize HC loss with each_class {}, pos_num {}, "
                    "sample_type {}, margin {} and beta {}."
                    .format(self.each_class, self.pos_num, self.sample_type, self.margin, self.beta))

    def _negative_threshold(self, pos_thres):
        """ this negative sampling strategy:
        sample_type:
            ratio: neg_num * neg_nn
            addative: pos_thres + neg_nn
            multiply: pos_thres * neg_nn
            original: neg_num
        """
        if self.sample_type == "original":
            neg_thres = 1e10
        elif self.sample_type == "addative":
            neg_thres = pos_thres + self.margin
        elif self.sample_type == "multiply":
            neg_thres = pos_thres * self.margin
        elif self.sample_type == "ratio":
            neg_thres = -1
        else:
            neg_thres = 1e10
        return neg_thres

    def _euclidean(self, feature):
        aa = torch.sum(feature * feature, dim=1).view(-1, 1)
        bb = aa.view(1, -1)
        ab = torch.matmul(feature, feature.t())
        dist = aa - 2*ab + bb
        return dist

    def forward(self, feature, label):
        feature = F.normalize(feature)
        batch_size, dim = feature.shape
        distance = self._euclidean(feature)
        loss = 0
        count = 0
        for i in range(batch_size):
            start = (i // self.each_class) * self.each_class
            end = start + self.each_class
            pos_index = torch.arange(start, end)
            # ignore distance=0
            ignore_index = pos_index != i
            pos_index = pos_index[ignore_index]
            neg_index = torch.cat([torch.arange(0, start), torch.arange(end, batch_size)])
            neg_num = len(neg_index)
            pos_dist = distance[i][pos_index]
            neg_dist = distance[i][neg_index]

            # sorted_pos_index: from small to large
            sorted_pos_index = torch.argsort(pos_dist)
            select_pos_index = sorted_pos_index[:self.pos_num]
            # threshold index for positive distance
            thres_pos_index = select_pos_index[-1]
            pos_thres = pos_dist[thres_pos_index]
            # threshold is also the max distance of positive pairs.
            max_pos_dist = pos_thres
            # calculate the within-class phi.
            pos_phi = torch.exp(pos_dist[select_pos_index] - max_pos_dist)
            within_phi = pos_phi.mean()

            neg_thres = self._negative_threshold(pos_thres)
            if self.sample_type == "ratio":
                sorted_neg_index = torch.argsort(neg_dist)
                thres_neg_num = int(neg_num * self.margin)
                thres_neg_index = sorted_neg_index[thres_neg_num-1]
                neg_thres = neg_dist[thres_neg_index]
            min_neg_dist = neg_dist.min()
            if neg_thres < min_neg_dist:
                neg_thres = min_neg_dist
            select_neg_index = neg_dist <= neg_thres
            neg_phi = torch.exp(min_neg_dist - neg_dist[select_neg_index])
            between_phi = neg_phi.mean()

            factor = torch.tensor(self.beta, device=feature.device, requires_grad=False)
            if max_pos_dist - min_neg_dist < self.max_gap:
                exp_adj = torch.exp(max_pos_dist - min_neg_dist)
                loss_one = torch.log(1 + self.beta * within_phi * between_phi * exp_adj)/torch.log(1 + factor)
            else:
                # numerical issue
                loss_one = (torch.log(self.beta * within_phi * between_phi) +
                        (max_pos_dist - min_neg_dist))/torch.log(1 + factor)
            count += 1
            loss += loss_one

        loss = loss / count
        return loss


if __name__ == "__main__":
    init_logger()
    torch.manual_seed(123)
    loss_fn = HCLoss(5, pos_nn=0.8, sample_type="multiply", margin=0.1)
    feature = torch.rand(20, 100)
    loss = loss_fn(feature)
    print("loss:", loss)