import math
from munch import munchify
from torchvision import transforms
import torch
from PIL import Image


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR


class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, opts, scale=0.875, random_crop=False,
                 random_hflip=False, random_vflip=False,
                 preserve_aspect_ratio=True):
        if type(opts) == dict:
            opts = munchify(opts)
        self.input_size = opts.input_size
        self.input_space = opts.input_space
        self.input_range = opts.input_range
        self.mean = opts.mean
        self.std = opts.std
        self.five_crop = opts.five_crop
        self.ten_crop = opts.ten_crop
        self.is_train = random_crop or random_hflip or random_vflip

        # https://github.com/tensorflow/models/blob/master/research/inception/inception/image_processing.py#L294
        self.scale = scale
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.random_crop = random_crop
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip

        if self.is_train and (self.five_crop or self.ten_crop):
            raise ValueError("Can not use five or ten crops when training.")

        if self.five_crop or self.ten_crop:
            # only used when testing
            self._init_multi_crop()
        else:
            # can use it when training or testing.
            self._init_single_crop()

    def _init_multi_crop(self):
        tfs = []
        if self.preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale)),
                                         interpolation=_pil_interp("bilinear")))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width),
                                         interpolation=_pil_interp("bilinear")))
        if self.ten_crop is True:
            tfs.append(transforms.TenCrop(max(self.input_size)))
        else:
            tfs.append(transforms.FiveCrop(max(self.input_size)))
        local_tfs = []
        local_tfs.append(transforms.ToTensor())
        local_tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        local_tfs.append(ToRange255(max(self.input_range)==255))
        local_tfs.append(transforms.Normalize(mean=self.mean, std=self.std))
        local_tfs = transforms.Compose(local_tfs)

        tfs.append(transforms.Lambda(lambda crops: torch.stack(
            [local_tfs(crop) for crop in crops])))

        self.tf = transforms.Compose(tfs)

    def _init_single_crop(self):
        tfs = []
        if self.preserve_aspect_ratio:
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size)/self.scale)),
                                         interpolation=_pil_interp("bilinear")))
        else:
            height = int(self.input_size[1] / self.scale)
            width = int(self.input_size[2] / self.scale)
            tfs.append(transforms.Resize((height, width),
                                         interpolation=_pil_interp("bilinear")))

        if self.random_crop:
            tfs.append(transforms.RandomCrop(max(self.input_size)))
        else:
            tfs.append(transforms.CenterCrop(max(self.input_size)))

        if self.random_hflip:
            tfs.append(transforms.RandomHorizontalFlip())

        if self.random_vflip:
            tfs.append(transforms.RandomVerticalFlip())

        tfs.append(transforms.ToTensor())
        tfs.append(ToSpaceBGR(self.input_space=='BGR'))
        tfs.append(ToRange255(max(self.input_range)==255))
        tfs.append(transforms.Normalize(mean=self.mean, std=self.std))

        self.tf = transforms.Compose(tfs)

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor

    def __repr__(self):
        return "\n===Attributes===:\n" \
               "size:" + str(self.input_size) + "\n" + "range:" + str(self.input_range) + "\n" \
               "color space:" + str(self.input_space) + "\n" \
               "mean:" + str(self.mean) + "\n" + "std:" + str(self.std) + "\n" \
               "===Operation===:\n" \
               "scale:" + str(self.scale) + "\n" \
               "random_crop:" + str(self.random_crop) + "\n" \
               "random_hflip:" + str(self.random_hflip) + "\n" \
               "random_vflip:" + str(self.random_vflip) + "\n"

