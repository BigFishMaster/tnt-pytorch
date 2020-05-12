from tnt.utils.misc import calc_topk_2d, calc_topk_multilabel
from tnt.utils.misc import calc_metric_topk
from tnt.utils.statistics import Statistics
from functools import partial


class Metric:
    def __init__(self, config):
        self.metric = config["name"]
        if self.metric == "acc_topk":
            topk = config["acc_topk"]
            self.fn = partial(calc_topk_2d, topk=topk)
            self.stats = partial(Statistics, topk=topk)
        elif self.metric == "multilabel_topk":
            topk = config["acc_topk"]
            self.fn = partial(calc_topk_multilabel, topk=topk)
            self.stats = partial(Statistics, topk=topk)
        elif self.metric == "metric_topk":
            topk = config["metric_topk"]
            self.fn = partial(calc_metric_topk, topk=topk)
            self.stats = partial(Statistics, topk=topk)
        else:
            raise NotImplementedError("{} is not implemented.".format(self.metric))

    def __call__(self, output, target, loss):
        vals = self.fn(output, target)
        stats = self.stats(loss=loss, n=target.size(0), vals=vals)
        return stats

