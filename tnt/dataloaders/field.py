import json
from functools import partial
from tnt.transform_image import TransformImage
from tnt.utils.logging import logger
from tnt.utils.io import *


__FORMAT__ = ["txt", "json"]
__MODAL__ = ["image", "text", "audio", "video", "label", "multi_label"]
__TYPE__ = ["path", "npy", "int"]


def check(v, c):
    if isinstance(v, list):
        return all(t in c for t in v)
    return v in c


class Field:
    def __init__(self, fmt, modals, types, opts, data_prefix=None, mode="train"):
        self.format = fmt
        self.modals = modals
        self.types = types
        self._getter = None
        if self.format == "txt":
            self._getter = load_txt
        elif self.format == "json":
            self._getter = partial(load_json, modals=self.modals)
        else:
            raise NotImplementedError("use {} for field_fn failed.".format(self.format))

        # image transforms
        if mode == "train":
            self.transforms = TransformImage(opts, random_crop=True, random_hflip=True)
        else:
            # valid and test
            self.transforms = TransformImage(opts)

        logger.info("In mode %s, image transforms are: %s", mode, self.transforms)

        self._fns = []
        for m, t in zip(self.modals, self.types):
            if m == "image" and t == "path":
                fn = partial(load_image_from_path, data_prefix=data_prefix, transforms=self.transforms)
            elif m == "image" and t == "npy":
                fn = partial(load_image_from_npy, data_prefix=data_prefix, transforms=self.transforms)
            elif m == "label" and t == "int":
                fn = lambda x: int(x)
            elif m in ["multi_label", "multi-label"] and t == "int":
                fn = lambda x: [int(i) for i in x]
            else:
                raise NotImplementedError("modals: {}, type: {} not implemented.".format(m, t))
            self._fns.append(fn)

    def _parser(self, fields, last):
        if last:
            return [self._fns[-1](fields[-1])]

        data = []
        for i, f in enumerate(fields):
            data.append(self._fns[i](f))
        return data

    def __call__(self, data, last=False):
        fields = self._getter(data)
        if len(fields) > 2:
            fields = [fields[0], fields[1:]]
        return self._parser(fields, last=last)

    @classmethod
    def from_cfg(cls, cfg, mode="train"):
        fmt = cfg["format"]
        assert check(fmt, __FORMAT__), "data format {} is not supported in {}.".format(fmt, __FORMAT__)
        modals = cfg["modals"]
        assert check(modals, __MODAL__), "data modal {} is not supported in {}.".format(modals, __MODAL__)
        types = cfg["types"]
        assert check(types, __TYPE__), "data type {} is not supported in {}.".format(types, __TYPE__)
        data_prefix = cfg["data_prefix"]
        opts = cfg["opts"]
        return cls(fmt, modals, types, opts, data_prefix, mode)
