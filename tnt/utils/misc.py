import numpy as np
import torch
import torch.nn.functional as F


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def calc_topk_2d(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        if isinstance(target, list):
            target = target[-1]
        if target.dim() == 2:
            target = target[:, 0]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def calc_topk_multilabel(output, target, topk=(1,)):
    """Computes the accuracy (hit-rate) over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        # pred: batch_size x maxk
        pred = pred.t()
        # target: batch_size x K
        batch_size, K = target.shape
        correct = torch.zeros_like(pred)
        for i in range(K):
            c = pred.eq(target[:, i].view(1, -1).expand_as(pred))
            correct += c.long()

        res = []
        for k in topk:
            topk_correct = correct[:k].sum(0)
            flag = topk_correct > 1
            topk_correct[flag] = 1
            correct_k = topk_correct.view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def calc_metric_topk(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size, _ = output.shape
        each_class = max(topk)
        fea_norm = F.normalize(output)
        cos_dist = torch.matmul(fea_norm, fea_norm.t())
        indices = range(batch_size)
        cos_dist[indices, indices] = -1
        gt_num = each_class - 1
        _, pred = cos_dist.topk(each_class, 1, True, True)
        pred = pred.cpu().numpy()
        starts = np.arange(batch_size // each_class).repeat(each_class) * each_class
        correct = np.zeros((batch_size, each_class))
        for i in range(batch_size):
            start = starts[i]
            rng = range(start, start + each_class)
            bin = [(p in rng) for p in pred[i]]
            correct[i] = bin

        res = []
        for k in topk:
            correct_k = correct[:, :k].sum() * 1.0 / gt_num
            res.append(correct_k)
        return res
