import os
import sys
import torch
from tnt.utils.io import load_model_from_file
from tnt.utils.logging import logger
import tnt.pretrainedmodels as pretrainedmodels
import torch.nn as nn


class ModelImpl:
    """ Initialize a classification model with model name and number of classes.

    Args:
        model_name_or_path (string): it can be a model name or a model path:

            * If it's a model name: the corresponding model will be selected from our pretrained model list.
              See :class:`~tnt.pretrainedmodels` for details.
            * If it's a model path: the model file pointed by the path should include a :func:`model()` function which
              implement a model class and return the instance.
        num_classes (int): the number of classes.
        pretrained (bool|string|None): whether to use pretrained models. It can be
            ``"imagenet"``, ``True`` or ``None``. Default: ``None``.
        gpu (int|None): specified ``gpu`` to run the model. It can be the ``gpu id`` for single-gpu training,
            or ``None`` for multi-gpu (all-available) training. Use `CUDA_VISIBLE_DEVICES` to
            control gpu number and ids. E.g., ``export CUDA_VISIBLE_DEVICES=0,2`` to run the model
            on gpu id ``0`` and ``2``. Default: ``None``.
    """
    def __init__(self, model_name_or_path, num_classes, pretrained=None, gpu=None):
        if os.path.exists(model_name_or_path):
            model_file = model_name_or_path
            model = load_model_from_file(model_file)
            if pretrained:
                state_dict = torch.load(pretrained)
                model.load_state_dict(state_dict)
        elif model_name_or_path in pretrainedmodels.model_names:
            model_name = model_name_or_path
            # input_space, input_size, input_range, mean, std
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
        else:  #  model.fc
            last_layer_name = "fc"
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            if out_features != num_classes:
                model.fc = nn.Linear(in_features, num_classes)

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
        """ Initialize the model from the configuration.

        Args:
            cls (:obj:`tnt.impls.ModelImpl`): The :class:`~tnt.impls.ModelImpl` class to instantiate.
            config (dict): a configuration to initialize the model.

        Returns:
            :obj:`torch.nn.Module`: an initialized model.

        """
        model_name = config["name"]
        pretrained = config["pretrained"]
        gpu = config["gpu"]
        loss_name = config.get("loss_name", None)
        if loss_name in ["HCLoss", "CosFaceLoss", "ArcFaceLoss", "MetricCELoss"]:
            num_classes = config["num_features"]
        else:
            num_classes = config["num_classes"]
        self = cls(model_name, num_classes, pretrained, gpu)
        return self.model
