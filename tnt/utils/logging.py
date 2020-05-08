# -*- coding: utf-8 -*-
from __future__ import absolute_import
import json
from functools import partial

import logging

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def beautify_info(info):
    if hasattr(info, "__dict__"):
        return json.dumps(info.__dict__, ensure_ascii=False, indent=4, default=str)
    else:
        return json.dumps(info, ensure_ascii=False, indent=4, default=str)


