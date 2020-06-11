import argparse
import json
import sys
import os
import yaml
import tnt.opts as opts
import tnt.utils.config as cfg
from tnt.utils.logging import init_logger, logger, beautify_info
from tnt.model_builder import ModelBuilder
import tnt.dataloaders as data_loaders

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config', default='', type=str, required=True, metavar='config path',
                    help='path to config file. (default: none)')


def create_config():
    opts.basic_opts(parser)
    args = parser.parse_args()
    config = cfg.init(args)
    return config


def runner(config):
    loader_type = config.get("dataloader_type", "general_data")
    if loader_type == "general_data":
        Loader = getattr(data_loaders, "GDLoader")
    elif loader_type == "metric_data":
        Loader = getattr(data_loaders, "MDLoader")
    else:
        logger.error("loader_type {} is not implemented.".format(loader_type))
        return

    builder = ModelBuilder(config)

    train_iter = Loader.from_config(cfg=config["data"], mode="train")
    valid_iter = Loader.from_config(cfg=config["data"], mode="valid")
    test_iter = Loader.from_config(cfg=config["data"], mode="test")

    is_train = "train" in config["data"]["mode"] and train_iter is not None
    is_valid = "valid" in config["data"]["mode"] and valid_iter is not None
    is_test = "test" in config["data"]["mode"] and test_iter is not None

    # TODO: config may be updated in data loader, e.g.: samples_per_class
    builder.update(config)

    if is_test:
        builder.run(test_iter=test_iter)
        return

    if is_valid and not is_train:
        builder.run(valid_iter=valid_iter)
        return

    logger.info('start training')
    builder.run(train_iter=train_iter, valid_iter=valid_iter)


def main():
    config = create_config()
    init_logger(config["log_file"])
    logger.info("create configure successfully: {}".format(beautify_info(config)))
    runner(config)


if __name__ == "__main__":
    main()
