import os
import sys
import json
from importlib import import_module
import torch
from PIL import Image
import numpy as np
from tnt.utils.logging import logger


def load_model_module(path):
    if not os.path.exists(path):
        raise ValueError("Model configuration not found in %s" % path)
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)
    sys.path.pop(0)

    if not hasattr(module, "model"):
        raise ImportError("No model defined in {}".format(path))

    return module


def load_model_from_file(path):
    module = load_model_module(path)
    model = module.model()
    del sys.path_importer_cache[os.path.dirname(module.__file__)]
    del sys.modules[module.__name__]
    return model


def save_checkpoint(state, filename, save_checkpoint_file=None):
    torch.save(state, filename)
    if save_checkpoint_file:
        open(save_checkpoint_file, "w").write(filename)


def load_checkpoint(checkpoint_file, gpu=None):
    if checkpoint_file.split(".")[-1] == "txt":
        checkpoint_file = open(checkpoint_file, "r").readline().strip()
    if os.path.isfile(checkpoint_file):
        if gpu is None:
            loc = None if torch.cuda.is_available() else torch.device("cpu")
            checkpoint = torch.load(checkpoint_file, map_location=loc)
        else:
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(checkpoint_file, map_location=loc)
        logger.info("load checkpoint from {}".format(checkpoint_file))
    else:
        logger.exception("no checkpoint file found in {}".format(checkpoint_file))
        return None
    return checkpoint


def load_txt(line):
    # split by "space" or "\t"
    fields = line.strip().split()
    return fields


def load_json(line, modals):
    dic1 = json.loads(line.strip())
    fields = [dic1[m] for m in modals]
    return fields


def load_image_from_path(path, data_prefix=None, transforms=None):
    if data_prefix is not None:
        path = data_prefix + path
    with open(path, "rb") as f:
        with Image.open(f) as img:
            img = img.convert("RGB")
    if transforms:
        img = transforms(img)
    return img


def load_image_from_path_box(path_box, data_prefix=None, box_extend=None, transforms=None):
    tts = path_box.split(",")
    path = tts[0]
    # min_x, min_y, max_x, max_y
    box = [int(t) for t in tts[1:]]
    if data_prefix is not None:
        path = data_prefix + path
    with open(path, "rb") as f:
        with Image.open(f) as img:
            img = img.convert("RGB")

    img_w, img_h = img.size
    min_x, min_y, max_x, max_y = box
    if box_extend:
        x1, y1, x2, y2 = [float(b) for b in box_extend.split(",")]
        w = max_x - min_x
        h = max_y - min_y
        new_min_x = max(0, min_x - x1 * w)
        new_max_x = min(img_w, max_x + x2 * w)
        new_min_y = max(0, min_y - y1 * h)
        new_max_y = min(img_h, max_y + y2 * h)
    else:
        new_min_x = max(0, min_x)
        new_max_x = min(img_w, max_x)
        new_min_y = max(0, min_y)
        new_max_y = min(img_h, max_y)

    img = img.crop([new_min_x, new_min_y, new_max_x, new_max_y])

    if transforms:
        img = transforms(img)
    return img


def load_image_from_npy(path, data_prefix=None, transforms=None):
    if data_prefix is not None:
        path = data_prefix + path
    img = np.load(path)
    if transforms:
        img = transforms(img)
    return img
