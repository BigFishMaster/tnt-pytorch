import torch
import tnt.losses as losses
from tnt.utils.logging import logger


class LossImpl:
    def __init__(self, loss_name, gpu, **kwargs):
        if loss_name == "ClassBalancedLoss":
            loss_type = kwargs.get("loss_type", "focal")
            beta = kwargs.get("classbalancedloss_beta", 0.9999)
            gamma = kwargs.get("classbalancedloss_gamma", 0.5)
            loss = losses.__dict__[loss_name](None, beta, gamma, loss_type)
        elif loss_name in ["CosFaceLoss", "ArcFaceLoss", "MultipleCosFaceLoss"]:
            num_features = kwargs["num_features"]
            num_classes = kwargs["num_classes"]
            scale = kwargs["arcface_scale"]
            margin = kwargs["arcface_margin"]
            logger.info("Initialize loss:{} with scale {} and margin {}".format(loss_name, scale, margin))
            loss = losses.__dict__[loss_name](num_features, num_classes, scale, margin)
        elif loss_name == "MetricCELoss":
            num_features = kwargs["num_features"]
            num_classes = kwargs["num_classes"]
            loss = losses.__dict__[loss_name](num_features, num_classes)
        elif loss_name == "HCLoss":
            each_class = kwargs["hc_each_class"]
            beta = kwargs["hc_beta"]
            pos_nn = kwargs["hc_pos_nn"]
            sample_type = kwargs["hc_sample_type"]
            margin = kwargs["hc_margin"]
            loss = losses.__dict__[loss_name](each_class, beta, pos_nn, sample_type, margin)
        else:
            loss = losses.__dict__[loss_name]()
        if loss_name in ["RelativeLabelLoss", "RelativeLabelLossV2"]:
            loss.gamma = kwargs.get("relativelabelloss_gamma", 0.2)
        if torch.cuda.is_available():
            loss = loss.cuda(gpu)
        self.loss = loss
        self.loss.name = loss_name
        self.gpu = gpu

    @classmethod
    def from_config(cls, config):
        loss_name = config["name"]
        gpu = config["gpu"]
        config.pop("name")
        config.pop("gpu")
        return cls(loss_name, gpu, **config).loss
