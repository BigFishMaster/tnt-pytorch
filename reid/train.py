# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
from collections import OrderedDict
from torch.backends import cudnn

from reid.config import cfg
from reid.data import make_data_loader
from reid.engine.trainer import do_train, do_train_with_center
from reid.modeling import build_model
from reid.layers import make_loss, make_loss_with_center
from reid.solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from reid.solver import make_optimizer_with_cosface_center

from reid.utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    num_classes = cfg.DATASETS.NUM_CLASSES
    train_loader, val_loader, num_query = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)

        loss_func = make_loss(cfg, num_classes)

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        cosface = None
        if cfg.MODEL.METRIC_LOSS_TYPE == "triplet_cosface_center":
            loss_func, center_criterion, cosface = make_loss_with_center(cfg, num_classes)
            optimizer, optimizer_center = make_optimizer_with_cosface_center(cfg, model, cosface, center_criterion)
        else:
            loss_func, center_criterion = make_loss_with_center(cfg, num_classes)
            optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
            print('Path to the checkpoint of center_param:', path_to_center_param)
            if cfg.MODEL.METRIC_LOSS_TYPE == "triplet_cosface_center":
                path_to_cosface_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'cosface_param')
                print('Path to the checkpoint of cosface_param:', path_to_cosface_param)
                cosface.load_state_dict(torch.load(path_to_cosface_param).state_dict())
            path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH).state_dict())
            optimizer.load_state_dict(torch.load(path_to_optimizer).state_dict())
            center_criterion.load_state_dict(torch.load(path_to_center_param).state_dict())
            optimizer_center.load_state_dict(torch.load(path_to_optimizer_center).state_dict())
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        elif cfg.MODEL.PRETRAIN_CHOICE == "weight":
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            new_state_dict = OrderedDict()
            for key in state_dict.keys():
                new_key = key.replace("module", "module.base")
                new_state_dict[new_key] = state_dict[key]
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print("loading model weights, missing_keys:{}, unexcepted_keys:{}".format(
                missing_keys, unexpected_keys))
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train_with_center(
            cfg,
            model,
            cosface,
            center_criterion,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,      # modify for using self trained model
            loss_func,
            num_query,
            start_epoch     # add for using self trained model
        )
    else:
        print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
