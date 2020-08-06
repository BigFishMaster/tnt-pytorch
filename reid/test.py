# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
from os import mkdir

import torch
from torch.backends import cudnn

from reid.config import cfg
from reid.data import make_inference_data_loader
from reid.engine.inference import inference
from reid.modeling import build_model
from reid.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    infer_loader = make_inference_data_loader(cfg)
    num_classes = cfg.DATASETS.NUM_CLASSES
    model = build_model(cfg, num_classes)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        model.module.load_param(cfg.TEST.WEIGHT)
    inference(cfg, model, infer_loader)


if __name__ == '__main__':
    main()
