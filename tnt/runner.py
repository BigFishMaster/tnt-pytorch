import argparse
import json
import sys
import os
import yaml
import tnt.opts as opts
from tnt.utils.logging import init_logger, logger, beautify_info
from tnt.model_builder import ModelBuilder
import tnt.dataloaders as data_loaders

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

    # initializing num_features
    if config["data"].get("num_features", None) is None:
        config["data"]["num_features"] = None
    if config["num_features"] is not None:
        config["data"]["num_features"] = config["num_features"]
    # end of initializing num_features
    # initializing num_classes
    if config["data"].get("num_classes", None) is None:
        config["data"]["num_classes"] = None
    if config["num_classes"] is not None:
        config["data"]["num_classes"] = config["num_classes"]
    # end of initializing num_classes
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
    #####################################################################
    config["data"]["pin_memory"] = config["disable_pin_memory"] is False
    config["data"]["sampler"]["use_first_label"] = config["use_first_label"]
    config["optimizer"]["fix_bn"] = config["fix_bn"]
    config["optimizer"]["fix_res"] = config["fix_res"]
    config["model"]["gpu"] = config["global"]["gpu"]
    config["model"]["loss_name"] = config["loss"]["name"]
    config["model"]["num_classes"] = config["data"]["num_classes"]
    config["model"]["num_features"] = config["data"]["num_features"]
    config["model"]["extract_feature"] = config["extract_feature"]
    config["model"]["multiple_pooling"] = config["multiple_pooling"]
    config["loss"]["gpu"] = config["global"]["gpu"]
    config["loss"]["relativelabelloss_gamma"] = config["relativelabelloss_gamma"]
    config["loss"]["num_classes"] = config["data"]["num_classes"]
    config["loss"]["num_features"] = config["data"]["num_features"]
    config["loss"]["arcface_scale"] = config["arcface_scale"]
    config["loss"]["arcface_margin"] = config["arcface_margin"]
    config["loss"]["hc_beta"] = config["hc_beta"]
    config["loss"]["hc_margin"] = config["hc_margin"]
    config["loss"]["hc_sample_type"] = config["hc_sample_type"]
    config["loss"]["hc_pos_nn"] = config["hc_pos_nn"]
    config["loss"]["hc_each_class"] = config["hc_each_class"]
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
