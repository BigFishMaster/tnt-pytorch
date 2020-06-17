import sys
import torch
from tnt.utils.logging import logger
import tnt.pretrainedmodels as pretrainedmodels


class FaceModelImpl:
    """ Initialize a face model with model name, number of features and input size.

    Args:
        model_name_or_path (string): the name of model to initialize. The corresponding model will be
            selected from our pretrained model list. See :class:`~tnt.pretrainedmodels` for details.
        num_features (int): the number of features.
        input_size (list[int]): a two-element list, which stands for the size of an input image.
        gpu (int|None): specified ``gpu`` to run the model. It can be the ``gpu id`` for single-gpu training,
            or ``None`` for multi-gpu (all-available) training. Use `CUDA_VISIBLE_DEVICES` to
            control gpu number and ids. E.g., ``export CUDA_VISIBLE_DEVICES=0,2`` to run the model
            on gpu id ``0`` and ``2``. Default: ``None``.
    """
    def __init__(self, model_name_or_path, num_features, input_size, gpu=None):
        if model_name_or_path in pretrainedmodels.model_names:
            model_name = model_name_or_path
            # input_space, input_size, input_range, mean, std
            model = pretrainedmodels.__dict__[model_name](input_size, num_features)
        else:
            logger.exception("'{}' is not available.".format(model_name_or_path))
            sys.exit()

        if gpu is not None:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        else:
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model).cuda()

        self.model = model
        self.model.last_layer_name = "BatchNorm"
        self.gpu = gpu

    @classmethod
    def from_config(cls, config):
        """ Initialize the model from the configuration.

        Args:
            cls (:obj:`tnt.impls.FaceModelImpl`): The :class:`~tnt.impls.FaceModelImpl` class to instantiate.
            config (dict): a configuration to initialize the model.

        Returns:
            :obj:`torch.nn.Module`: an initialized model.

        """
        model_name = config["name"]
        gpu = config["gpu"]
        num_features = config["num_features"]
        input_size = [config["image_size"], config["image_size"]]
        self = cls(model_name, num_features, input_size, gpu)
        return self.model
