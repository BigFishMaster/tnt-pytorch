import argparse
import sys
import os
import yaml
import tnt.opts as opts
from tnt.utils.logging import init_logger, logger
from tnt.model_builder import ModelBuilder
from tnt.dataloaders.data_loader import GeneralDataLoader as GDLoader

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--config', default='', type=str, required=True, metavar='config path',
                    help='path to config file. (default: none)')


def create_config():
    opts.basic_opts(parser)
    args = parser.parse_args()
    config = args.__dict__
    if args.config is None:
        logger.exception("Please specify config file by: \n"
                         "python train.py --config training.yml")
        sys.exit()
    else:
        yaml_config = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        config.update(yaml_config)
    config["model"]["gpu"] = config["global"]["gpu"]
    config["model"]["num_classes"] = config["data"]["num_classes"]
    config["loss"]["gpu"] = config["global"]["gpu"]
    log_dir = config["global"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config["log_file"] = os.path.join(log_dir, config["global"]["log_file"])
    os.environ["TORCH_HOME"] = config["model"]["TORCH_HOME"]
    logger.info("create configure successfully.")
    return config


def runner(config):
    builder = ModelBuilder(config)
    train_iter = GDLoader.from_config(cfg=config["data"], mode="train")
    valid_iter = GDLoader.from_config(cfg=config["data"], mode="valid")
    test_iter = GDLoader.from_config(cfg=config["data"], mode="test")
    is_train = "train" in config["data"]["mode"] and train_iter is not None
    is_valid = "valid" in config["data"]["mode"] and valid_iter is not None
    is_test = "test" in config["data"]["mode"] and test_iter is not None

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
    runner(config)


if __name__ == "__main__":
    main()
