import torch
import tnt.losses as losses


class LossImpl:
    def __init__(self, loss_name, gpu, **kwargs):
        if loss_name == "ClassBalancedLoss":
            loss_type = kwargs.get("loss_type", "focal")
            beta = kwargs.get("classbalancedloss_beta", 0.9999)
            gamma = kwargs.get("classbalancedloss_gamma", 0.5)
            loss = losses.__dict__[loss_name](None, beta, gamma, loss_type)
        elif loss_name in ["CosFaceLoss", "ArcFaceLoss"]:
            num_features = kwargs["num_features"]
            num_classes = kwargs["num_classes"]
            loss = losses.__dict__[loss_name](num_features, num_classes)
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
