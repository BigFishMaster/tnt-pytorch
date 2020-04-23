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
    # update parameters from args:
    if config["gpu"] is not None:
        config["global"]["gpu"] = config["gpu"]
    if config["num_classes"] is not None:
        config["data"]["num_classes"] = config["num_classes"]
    if config["pretrained"] is not None:
        config["model"]["pretrained"] = config["pretrained"]
    if config["model_name"] is not None:
        config["model"]["name"] = config["model_name"]
    if config["resume"] is not None:
        config["global"]["resume"] = config["resume"]
    if config["weight"] is not None:
        config["global"]["weight"] = config["weight"]
    if config["num_epochs"] is not None:
        config["global"]["num_epochs"] = config["num_epochs"]
    if config["batch_size"] is not None:
        config["data"]["sampler"]["batch_size"] = config["batch_size"]
    if config["mode"] is not None:
        config["data"]["mode"] = config["mode"].split(",")
    if config["test"] is not None:
        config["data"]["test"] = config["test"]
    if config["out_file"] is not None:
        if config["data"].get("output") is None:
            config["data"]["output"] = {}
        config["data"]["output"]["file"] = config["out_file"]
    if config["out_mode"] is not None:
        if config["data"].get("output") is None:
            config["data"]["output"] = {}
        config["data"]["output"]["mode"] = config["out_mode"]
    #####################################
    config["data"]["pin_memory"] = config["disable_pin_memory"] is False
    config["optimizer"]["fix_bn"] = config["fix_bn"]
    config["optimizer"]["fix_res"] = config["fix_res"]
    config["model"]["gpu"] = config["global"]["gpu"]
    config["model"]["num_classes"] = config["data"]["num_classes"]
    config["loss"]["gpu"] = config["global"]["gpu"]
    num_epochs = config["global"]["num_epochs"]
    config["lr_strategy"]["num_epochs"] = num_epochs
    # accumlation steps will affect the steps number of each epoch.
    accum_steps = config["optimizer"].get("accum_steps", 1)
    accum_steps = 1 if accum_steps is None else accum_steps
    steps_each_epoch = config["lr_strategy"].get("steps_each_epoch", 0)
    config["lr_strategy"]["steps_each_epoch"] = steps_each_epoch // accum_steps
    log_dir = config["global"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config["log_file"] = os.path.join(log_dir, config["global"]["log_file"])
    os.environ["TORCH_HOME"] = config["model"]["TORCH_HOME"]
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
    logger.info("create configure successfully: {}".format(config))
    runner(config)


if __name__ == "__main__":
    main()
