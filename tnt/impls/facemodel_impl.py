import sys
import torch
from tnt.utils.logging import logger
import tnt.pretrainedmodels as pretrainedmodels


class FaceModelImpl:
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
        model_name = config["name"]
        gpu = config["gpu"]
        num_features = config["num_features"]
        input_size = [config["image_size"], config["image_size"]]
        self = cls(model_name, num_features, input_size, gpu)
        return self.model
