from functools import partial
from tnt.data.transform_image import TransformImage
from tnt.data.transforms_factory import transforms_imagenet_train, transforms_imagenet_eval
from tnt.utils.io import *


__FORMAT__ = ["txt", "json"]
__MODAL__ = ["image", "label", "multi_label", "image_weight", "label_weight"]
# "none" is used to skip some fields.
__TYPE__ = ["path", "path_box", "npy", "int", "float", "pseudo", "none"]


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
        is_auto_augment = opts.transform_type in ["v0", "v0r", "original", "originalr"]
        if mode == "train":
            if is_auto_augment:
                self.transforms = transforms_imagenet_train(
                    img_size=(opts.input_size[1], opts.input_size[2]),
                    auto_augment=opts.transform_type,
                )
            else:
                self.transforms = TransformImage(opts, random_crop=True, random_hflip=True)
        else:
            # valid and test
            if is_auto_augment:
                self.transforms = transforms_imagenet_eval(
                    img_size=(opts.input_size[1], opts.input_size[2])
                )
            else:
                self.transforms = TransformImage(opts)

        logger.info("In mode %s, image transforms are: %s", mode, self.transforms)

        self._fns = []
        self._ignore_fields = []
        for i, (m, t) in enumerate(zip(self.modals, self.types)):
            if m == "image" and t == "path":
                fn = partial(load_image_from_path, data_prefix=data_prefix, transforms=self.transforms)
            elif m == "image" and t == "path_box":
                box_extend = opts.box_extend
                fn = partial(load_image_from_path_box, data_prefix=data_prefix, box_extend=box_extend,
                             transforms=self.transforms)
            elif m == "image" and t == "npy":
                fn = partial(load_image_from_npy, data_prefix=data_prefix, transforms=self.transforms)
            elif m == "label" and t == "int":
                fn = lambda x: int(x)
            elif m == "multi_label" and t == "int":
                def _func(x):
                    if isinstance(x, str):
                        x = [x]
                    return [int(i) for i in x]
                fn = _func
            elif m == "label_weight" and t == "float":
                fn = lambda x: float(x)
            elif m == "image_weight" and t == "float":
                fn = lambda x: float(x)
                self._ignore_fields.append(i)
            elif m == "label" and t == "pseudo":
                # e.g.: path l1:s1,l2:s2,l2:s3
                def _func(x):
                    tts = [t.split(":") for t in x.split(",")]
                    labels = [int(l) for l, _ in tts]
                    scores = [float(s) for _, s in tts]
                    return labels, scores
                fn = _func
            elif t == "none":
                fn = None
            else:
                raise NotImplementedError("modals: {}, type: {} not implemented.".format(m, t))
            self._fns.append(fn)

    def _parser(self, fields, last):
        if isinstance(last, list):
            return [self._fns[l](fields[l]) for l in last]
        elif isinstance(last, bool):
            if last:
                return [self._fns[-1](fields[-1])]
        # TODO: this is true if last is bool type.
        elif isinstance(last, int):
            return [self._fns[last](fields[last])]
        else:
            pass

        # warn: in test mode, only return the image array;
        #       the label is ommited although its _fn is initialized.
        data = []
        for i, f in enumerate(fields):
            if i in self._ignore_fields:
                continue
            if self._fns[i] is None:
                continue
            data.append(self._fns[i](f))
        return data

    def __call__(self, data, last=False):
        fields = self._getter(data)
        if len(fields) > len(self._fns):
            # TODO: only support N vs. 2, e.g. mulit-label classification
            #   to be extended to N vs. k (N > k), e.g. multi-task classification
            fields = [fields[0], fields[1:]]
        else:
            # support N < k and N == k.
            pass
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
