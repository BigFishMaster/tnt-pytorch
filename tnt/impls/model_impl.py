import os
import sys
import torch
from tnt.utils.io import load_model_from_file
from tnt.utils.logging import logger
import tnt.pretrainedmodels as pretrainedmodels
import torch.nn as nn
from tnt.layers import MultiPoolingModel


class ModelImpl:
    def __init__(self, model_name_or_path, num_classes, pretrained=None, gpu=None,
                 swish=False,
                 extract_feature=False, multiple_pooling=False, mp_layers="conv+relu"):
        if os.path.exists(model_name_or_path):
            model_file = model_name_or_path
            model = load_model_from_file(model_file)
            if pretrained:
                state_dict = torch.load(pretrained)
                model.load_state_dict(state_dict)
        elif model_name_or_path in pretrainedmodels.model_names:
            model_name = model_name_or_path
            # input_space, input_size, input_range, mean, std
            if multiple_pooling:
                model = MultiPoolingModel(model_name, num_classes, mp_layers, pretrained)
                logger.info("MultiPoolingModel with name: {} and feature: {}.".format(model_name, num_classes))
            elif extract_feature:
                kwargs = {"extract_feature": extract_feature}
                model = pretrainedmodels.__dict__[model_name](pretrained=pretrained, **kwargs)
            else:
                model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
            logger.info("model pretrained: %s", pretrained)
        else:
            logger.exception("'{}' is not available.".format(model_name_or_path))
            sys.exit()

        # TODO: fix the parameters.
        # e.g.
        # for param in model.parameters():
        #     param.requires_grad = False

        # TODO: initialize the last_linear layer
        # url: https://pytorch.org/docs/master/notes/autograd.html
        if extract_feature is False and multiple_pooling is False:
            # for efficientnet models:
            last_layer_name = None
            if hasattr(model, "classifier"):
                last_layer_name = "classifier"
                in_features = model.classifier.in_features
                out_features = model.classifier.out_features
                if out_features != num_classes:
                    model.classifier = nn.Linear(in_features, num_classes)
            # for torchvision models
            elif hasattr(model, "last_linear"):
                last_layer_name = "last_linear"
                in_features = model.last_linear.in_features
                out_features = model.last_linear.out_features
                if out_features != num_classes:
                    model.last_linear = nn.Linear(in_features, num_classes)
            # for billionscale models
            elif hasattr(model, "fc"):  #  model.fc
                last_layer_name = "fc"
                in_features = model.fc.in_features
                out_features = model.fc.out_features
                if out_features != num_classes:
                    model.fc = nn.Linear(in_features, num_classes)
            # for efficienetnet_pytorch
            elif hasattr(model, "_fc"):
                last_layer_name = "fc"
                in_features = model._fc.in_features
                out_features = model._fc.out_features
                if out_features != num_classes:
                    model._fc = nn.Linear(in_features, num_classes)
            # for squeezenet
            elif hasattr(model, "last_conv"):
                last_layer_name = "last_conv"
                in_features = model.last_conv.in_channels
                out_features = model.last_conv.out_channels
                if out_features != num_classes:
                    model.last_conv = nn.Conv2d(in_features, num_classes, kernel_size=1)
        else:
            last_layer_name = "none"

        if gpu is not None:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        else:
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model).cuda()

        self.model = model
        self.model.last_layer_name = last_layer_name
        self.gpu = gpu

    @classmethod
    def from_config(cls, config):
        model_name = config["name"]
        pretrained = config["pretrained"]
        gpu = config["gpu"]
        loss_name = config.get("loss_name", None)
        if loss_name in ["HCLoss", "CosFaceLoss", "ArcFaceLoss", "MetricCELoss", "MultipleCosFaceLoss"]:
            num_classes = config["num_features"]
        else:
            num_classes = config["num_classes"]
        extract_feature = config["extract_feature"]
        multiple_pooling = config["multiple_pooling"]
        mp_layers = config["mp_layers"]
        self = cls(model_name, num_classes, pretrained, gpu, extract_feature,
                   multiple_pooling, mp_layers)
        return self.model
