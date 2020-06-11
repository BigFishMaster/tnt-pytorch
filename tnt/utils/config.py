from collections import namedtuple
import yaml
import os


def init_opts(config):
    opts = namedtuple("opts", [
        "input_space", "input_size", "input_range", "mean", "std",
        "five_crop", "ten_crop", "transform_type", "image_scale",
        "preserve_aspect_ratio", "random_erase", "random_crop",
        "random_hflip", "box_extend"
    ])
    opts.five_crop = config["five_crop"]
    opts.ten_crop = config["ten_crop"]
    opts.transform_type = config["transform_type"]
    opts.image_scale = config["image_scale"]
    opts.preserve_aspect_ratio = config["preserve_aspect_ratio"] != 0
    opts.random_erase = config["enable_random_erase"]
    opts.random_crop = config["disable_random_crop"] is False
    opts.random_hflip = config["disable_random_hflip"] is False
    opts.box_extend = config["box_extend"]
    config["data"]["opts"] = opts
    return opts


def init(args):
    config = args.__dict__
    if args.config is None:
        raise ValueError("Please specify config file:\n tnt_runner --config training.yml")
    else:
        yaml_config = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        config.update(yaml_config)

    # update hyper-parameters from args:
    """Update Global Opts"""
    if config["gpu"] is not None:
        config["global"]["gpu"] = config["gpu"]
    if config["resume"] is not None:
        config["global"]["resume"] = config["resume"]
    if config["weight"] is not None:
        config["global"]["weight"] = config["weight"]
    if config["num_epochs"] is not None:
        config["global"]["num_epochs"] = config["num_epochs"]

    """Update Data Opts"""
    if config["data"].get("num_features", None) is None:
        config["data"]["num_features"] = None
    if config["num_features"] is not None:
        config["data"]["num_features"] = config["num_features"]
    if config["batch_size"] is not None:
        config["data"]["sampler"]["batch_size"] = config["batch_size"]
    if config["mode"] is not None:
        config["data"]["mode"] = config["mode"].split(",")
    if config["data"].get("num_classes", None) is None:
        config["data"]["num_classes"] = None
    if config["num_classes"] is not None:
        config["data"]["num_classes"] = config["num_classes"]
    config["data"]["pin_memory"] = config["disable_pin_memory"] is False
    config["data"]["sampler"]["use_first_label"] = config["use_first_label"]

    """Update Model Opts"""
    if config["pretrained"] is not None:
        config["model"]["pretrained"] = config["pretrained"]
    if config["model_name"] is not None:
        config["model"]["name"] = config["model_name"]
    config["model"]["fix_bn"] = config["fix_bn"]
    config["model"]["fix_res"] = config["fix_res"]
    config["model"]["gpu"] = config["global"]["gpu"]
    config["model"]["loss_name"] = config["loss"]["name"]
    config["model"]["num_classes"] = config["data"]["num_classes"]
    config["model"]["num_features"] = config["data"]["num_features"]
    # it will be used in facemodelimpl.
    config["model"]["image_size"] = config["image_size"] if config["image_size"] else 224

    """Update Test Opts"""
    if config["test"] is not None:
        config["data"]["test"] = config["test"]
    if config["data"].get("output") is None:
        config["data"]["output"] = {}
        config["data"]["output"]["mode"] = "top5"
        config["data"]["output"]["file"] = config["global"]["log_dir"] + "test.output"
    if config["out_mode"] is not None:
        config["data"]["output"]["mode"] = config["out_mode"]
    if config["out_file"] is not None:
        config["data"]["output"]["file"] = config["out_file"]

    """Update Loss Opts"""
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

    """Update Learning Rate Opts"""
    config["lr_strategy"]["num_epochs"] = config["global"]["num_epochs"]

    """Update Optim Opts"""
    accum_steps = config["optimizer"].get("accum_steps", 1)
    accum_steps = 1 if accum_steps is None else accum_steps
    config["optimizer"]["accum_steps"] = accum_steps
    # accumlation steps will affect the steps number of each epoch.
    steps_each_epoch = config["lr_strategy"].get("steps_each_epoch", 0)
    config["lr_strategy"]["steps_each_epoch"] = steps_each_epoch // accum_steps

    """Update Transform Opts"""
    init_opts(config)

    """Log and Torch ENV."""
    log_dir = config["global"]["log_dir"]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config["log_file"] = os.path.join(log_dir, config["global"]["log_file"])
    os.environ["TORCH_HOME"] = config["model"]["TORCH_HOME"]
    return config
