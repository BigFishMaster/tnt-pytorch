import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tnt.losses.utils import generate_index
from tnt.utils.logging import logger, init_logger


class MetricCELoss(nn.Module):
    """ Cross-entropy loss for metric learning with a specified feature size.
    In addition, there exists a ReLU layer to pre-process the input feature.

    Args:
        feature_size (int): usually 128, 256, 512 ...
        num_classes (int): num of classes when training
    """

    def __init__(self, feature_size, num_classes):
        super(MetricCELoss, self).__init__()
        self.in_features = feature_size
        self.out_features = num_classes
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

    def output(self, feature):
        """ Output the logit.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.

        Returns:
            :obj:`torch.FloatTensor`: logit
        """
        output = F.linear(F.relu(feature), self.weight)
        return output

    def forward(self, feature, label):
        """ Calculate MetricCE loss.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.
            label (:obj:`torch.LongTensor`): image labels.

        Returns:
            (:obj:`torch.FloatTensor`, :obj:`torch.FloatTensor`):

            * loss: MetricCE loss over the batch.
            * logit: logit output after ReLU.
        """
        output = self.output(feature)
        loss = self.ce(output, label)
        return loss, output


class HCLoss(nn.Module):
    """ The implementation of Loss in the paper:"Large Scale Strongly Supervised Ensemble Metric Learning,
    with Applications to Face Verification and Retrieval"(https://arxiv.org/abs/1212.6094)

    Args:
        each_class (int): sample number of each class.
        beta (float): loss scale. Default: ``10000.0``
        pos_nn (float): the ratio of nearest neighbour positive samples. Default: ``1.0``.
        sample_type (float): negative sampling strategy. Default: ``ratio``. It can be

            * ``original``: :math:`neg\_num`
            * ``addative``: :math:`pos\_thres + margin`
            * ``multiply``: :math:`pos\_thres * margin`
            * ``ratio``: :math:`neg\_num * margin`
        margin: the margin to determine the negative samples. Default: ``1.0``.
    """

    def __init__(self, each_class, beta=10000.0, pos_nn=1.0, sample_type="ratio", margin=1.0):
        super(HCLoss, self).__init__()
        self.each_class = each_class
        self.beta = beta
        self.pos_num = int(pos_nn * (each_class - 1))
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
        dist = aa - 2 * ab + bb
        return dist

    def forward(self, feature, label):
        """ Calculate HC loss.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.
            label (:obj:`torch.LongTensor`): image labels.

        Returns:
            :obj:`torch.FloatTensor`: The MetricCE loss over the batch.
        """
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
                thres_neg_index = sorted_neg_index[thres_neg_num - 1]
                neg_thres = neg_dist[thres_neg_index]
            min_neg_dist = neg_dist.min()
            if neg_thres < min_neg_dist:
                neg_thres = min_neg_dist
            select_neg_index = neg_dist <= neg_thres
            neg_phi = torch.exp(min_neg_dist - neg_dist[select_neg_index])
            between_phi = neg_phi.mean()

            factor = torch.tensor(self.beta, device=feature.device)
            if max_pos_dist - min_neg_dist < self.max_gap:
                exp_adj = torch.exp(max_pos_dist - min_neg_dist)
                loss_one = torch.log(1 + self.beta * within_phi * between_phi * exp_adj) / torch.log(1 + factor)
            else:
                # numerical issue
                loss_one = (torch.log(self.beta * within_phi * between_phi) +
                            (max_pos_dist - min_neg_dist)) / torch.log(1 + factor)
            count += 1
            loss += loss_one

        loss = loss / count
        return loss


class HCLossV2(nn.Module):
    """ The implementation of Loss in the paper:"Large Scale Strongly Supervised Ensemble Metric Learning,
    with Applications to Face Verification and Retrieval"(https://arxiv.org/abs/1212.6094)

    Args:
        each_class (int): sample number of each class.
        beta (float): loss scale. Default: ``10000.0``
        pos_nn (float): the ratio of nearest neighbour positive samples. Default: ``1.0``.
        sample_type (str): negative sampling strategy. Default: ``ratio``. It can be

            * ``original``: :math:`neg\_num`
            * ``addative``: :math:`pos\_thres + margin`
            * ``multiply``: :math:`pos\_thres * margin`
            * ``ratio``: :math:`neg\_num * margin`
        margin: the margin to determine the negative samples. Default: ``1.0``.
    """

    def __init__(self, batch_size, each_class, beta=100000.0, pos_nn=1.0, sample_type="ratio", margin=1.0):
        super(HCLossV2, self).__init__()
        self.each_class = each_class
        self.beta = beta
        self.pos_num = int(pos_nn * (each_class - 1))
        self.sample_type = sample_type
        self.margin = margin
        self.max_gap = 20
        self.log_num = 0
        # add buffer to deal with index order.
        select_index = generate_index(batch_size, each_class)
        self.register_buffer("select_index", select_index)
        logger.info("Initialize HC loss with each_class {}, pos_num {}, "
                    "sample_type {}, margin {} and beta {}."
                    .format(self.each_class, self.pos_num, self.sample_type, self.margin, self.beta))

    def _negative_threshold(self, pos_thres):
        """ this negative sampling strategy:
        sample_type:
            addative: pos_thres + neg_nn
            multiply: pos_thres * neg_nn
            original: neg_num
        """
        if self.sample_type == "original":
            neg_thres = torch.ones_like(pos_thres) * 1e10
        elif self.sample_type == "addative":
            neg_thres = pos_thres + self.margin
        elif self.sample_type == "multiply":
            neg_thres = pos_thres * self.margin
        else:
            neg_thres = torch.ones_like(pos_thres) * 1e10
        return neg_thres

    def _euclidean(self, feature):
        # aa: shape=(batch_size, 1)
        aa = torch.sum(feature * feature, dim=1).view(-1, 1)
        bb = aa.view(1, -1)
        ab = torch.matmul(feature, feature.t())
        dist = aa - 2 * ab + bb
        # distance shape is: [batch_size*batch_size]
        dist = dist.reshape(-1)
        return dist

    def forward(self, feature, label=None):
        """ Calculate HC loss.

        Args:
            feature (:obj:`torch.FloatTensor`): image features from previous layers.
            label (:obj:`torch.LongTensor`): image labels.

        Returns:
            :obj:`torch.FloatTensor`: The MetricCE loss over the batch.
        """
        batch_size, dim = feature.shape
        # calculate the Euclidean distance of each pair.
        distance = self._euclidean(feature)
        # re-order the values of distance as self-tensor, positive-tensor, negative-tensor.
        distance = torch.gather(distance, 0, self.select_index)
        # reshape the distance to shape: batch_size x batch_size
        distance = distance.reshape(batch_size, batch_size)
        # split distance to obtain postive-tensor and negative-tensor.
        ignore, pos, neg = torch.split(distance, [1, self.each_class-1, batch_size-self.each_class], dim=1)

        """hard sampling"""
        # sort positive-tensor in ascending order.
        sorted_pos, _ = torch.sort(pos, dim=1)
        # select the positive threshold, beyond which positives are outliers.
        pos_thres = sorted_pos[:, self.pos_num-1].reshape(-1, 1)
        # normalize positive distances while skipping outliers.
        pos_phi = torch.where(pos<=pos_thres,
                              torch.exp(pos-pos_thres),
                              torch.zeros_like(pos))
        # calculate mean positive distance with available positive distances.
        real_pos_num = (pos_phi > 0).sum(dim=1, keepdim=True)
        real_pos_num = real_pos_num.float()
        pos_mean = torch.sum(pos_phi, dim=1, keepdim=True) / real_pos_num

        # calculate neg_thres according to pos_thres nad negative sampling strategy.
        neg_thres = self._negative_threshold(pos_thres)
        # calculate neg_min
        neg_min, _ = neg.min(dim=1, keepdim=True)
        neg_max, _ = neg.max(dim=1, keepdim=True)
        # update neg_thres with neg_min: at least one negative distance is included.
        neg_thres = torch.where(neg_thres<neg_min, neg_min, neg_thres)
        # update neg_thres with neg_max: at most all negative distances are included.
        neg_thres = torch.where(neg_thres>neg_max, neg_max, neg_thres)
        # normalize negative distances while skipping outliers.
        # neg_thres is used to select negative pairs, and
        # neg_min is used to control numerical problem.
        neg_phi = torch.where(neg<=neg_thres,
                              torch.exp(neg_min-neg),
                              torch.zeros_like(neg))
        real_neg_num = (neg_phi > 0).sum(dim=1, keepdim=True)
        real_neg_num = real_neg_num.float()
        neg_mean = torch.sum(neg_phi, dim=1, keepdim=True) / real_neg_num

        bias = pos_thres - neg_min
        theta = pos_mean * neg_mean
        loss = self._surrogate_function(bias, theta)
        self.log_num += 1
        if self.log_num % 100 == 0:
            self.log_num = 0
            logger.info("pos:{}\npos_thres:{}\npos_phi:{}\nreal_pos_num:{}\npos_mean:{}\n"
                        "neg:{}\nneg_thres:{}\nneg_phi:{}\nreal_neg_num:{}\nneg_mean:{}\n"
                        "bias:{}\ntheta:{}\nloss:{}\n".format(
                pos, pos_thres, pos_phi, real_pos_num, pos_mean,
                neg, neg_thres, neg_phi, real_neg_num, neg_mean,
                bias, theta, loss))
        return loss

    def _surrogate_standard(self, bias, theta):
        output = torch.log(1 + self.beta * theta * torch.exp(bias))/math.log(1 + self.beta)
        return output

    def _surrogate_approximate(self, bias, theta):
        output = (torch.log(theta) + bias + math.log(self.beta))/math.log(1 + self.beta)
        return output

    def _surrogate_function(self, bias, theta):
        """
        Args:
            bias: bias to deal with overflow problem.
            theta: product of inner- and inter-class statistics.

        Returns:

        """
        loss = torch.where(
            bias > self.max_gap,
            self._surrogate_approximate(bias, theta),
            self._surrogate_standard(bias, theta)
        )
        return loss


if __name__ == "__main__":
    init_logger()
    torch.manual_seed(123)
    loss_fn = HCLossV2(batch_size=20, each_class=5, pos_nn=0.8, sample_type="multiply", margin=1.2)
    feature = torch.rand(20, 100)
    loss = loss_fn(feature)
    print("loss:", loss)
