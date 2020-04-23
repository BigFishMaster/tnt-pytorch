import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from tnt.utils.logging import logger
from tnt.utils.collate_fn import multilabel_collate_fn
from tnt.dataloaders.field import Field


class GeneralDataLoader(Dataset):
    def __init__(self, cfg, mode="train"):
        self.num_classes = cfg["num_classes"]
        filename = cfg[mode]
        self.data_list = open(filename, "r", encoding="utf8").readlines()
        self.data_list = np.array(self.data_list, dtype=np.string_)
        logger.info("In mode {}, data_list has length of {}.".format(mode, len(self.data_list)))

        # field processor
        self._field = Field.from_cfg(cfg, mode=mode)

        # sampling strategy
        sampler_config = cfg["sampler"]
        self.sampler = self._create_sampler(sampler_config, mode)
        self.collate_fn = self._create_collate(sampler_config)
        self.batch_size = sampler_config.get("batch_size", 10)
        self.num_workers = sampler_config.get("num_workers", 4)

    @classmethod
    def from_config(cls, cfg, mode="train"):
        if (mode not in cfg) or (not cfg[mode]):
            return None
        self = cls(cfg, mode)
        logger.info("===mode-{}===\ncollate_fn: {}\nsampler:{}".format(
            mode, self.collate_fn, self.sampler))
        is_shuffle = True
        if mode != "train" or self.sampler is not None:
            is_shuffle = False
        pin_memory = cfg["pin_memory"]
        data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=is_shuffle,
                                 num_workers=self.num_workers, collate_fn=self.collate_fn,
                                 pin_memory=pin_memory, sampler=self.sampler, drop_last=(mode == "train"))
        logger.info("data loader is: {}".format(data_loader.__dict__))
        return data_loader

    def __getitem__(self, index):
        data = self.data_list[index].decode()
        # there are two cases:
        # train or valid:
        # 1. normal:       image, label
        # 2. multi-label:  image, [label1, label2]
        # 3. weight-label: image, score, label
        # test:
        # 1. image
        result = self._field(data)
        return result

    def __len__(self):
        return len(self.data_list)

    def _create_collate(self, cfg):
        strategy = cfg.get("strategy")
        if strategy == "multilabel_balanced":
            return multilabel_collate_fn
        else:
            return default_collate

    def _create_sampler(self, cfg, mode):
        if mode != "train":
            return None
        strategy = cfg.get("strategy")
        logger.info("In mode {}, sampling strategy is {}".format(mode, strategy))
        num_samples = cfg.get("num_samples") or len(self.data_list)
        replacement = cfg.get("replacement") or True
        sample_labels = []
        if strategy == "class_balanced":
            for i, data in enumerate(self.data_list):
                if i % 10000 == 0:
                    logger.info("creating sampler for data: %d/%d", i, len(self.data_list))
                label = self._field(data.decode(), last=True)
                sample_labels.append(label[0])
            logger.info("creating sampler totally: %d", len(self.data_list))

            class_weights = Counter()
            class_weights.update(sample_labels)
            if len(class_weights) != self.num_classes:
                logger.warning("{} classes have no samples.".format(self.num_classes-len(class_weights)))
                for cls in range(self.num_classes):
                    if cls not in class_weights:
                        class_weights.update({cls:1e9})

            sorted_weights = sorted(class_weights.items(), key=lambda x: x[0])
            weights = [s[1] for s in sorted_weights]
            weights = 1.0 / np.array(weights)
            sample_weights = weights[sample_labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples,
                                            replacement=replacement)
            del weights
            del sample_weights
            del sample_labels
            del class_weights
            del sorted_weights
            return sampler
        elif strategy == "instance_weighted":
            sample_weights = np.zeros(len(self.data_list))
            for i, data in enumerate(self.data_list):
                if i % 10000 == 0:
                    logger.info("creating sampler for data: %d/%d", i, len(self.data_list))
                # data is [image, score, label]
                score, label = self._field(data.decode(), last=[1, 2])
                sample_weights[i] = score
                sample_labels.append(label)
            logger.info("creating sampler totally: %d", len(self.data_list))

            class_weights = Counter()
            class_weights.update(sample_labels)
            if len(class_weights) != self.num_classes:
                logger.warning("{} classes have no samples.".format(self.num_classes-len(class_weights)))
                for cls in range(self.num_classes):
                    if cls not in class_weights:
                        class_weights.update({cls:1e9})
            sorted_weights = sorted(class_weights.items(), key=lambda x: x[0])
            class_weights = [s[1] for s in sorted_weights]
            class_weights = 1.0 / np.array(class_weights)
            class_weights = class_weights[sample_labels]
            weights = sample_weights * class_weights
            sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                            replacement=replacement)
            del weights
            del sample_weights
            del sample_labels
            del class_weights
            del sorted_weights
            return sampler
        elif strategy == "multilabel_balanced":
            class_weights = Counter()
            for i, data in enumerate(self.data_list):
                if i % 100000 == 0:
                    logger.info("creating sampler for data: %d/%d", i, len(self.data_list))
                label = self._field(data.decode(), last=True)
                # label[0] is a list
                sample_labels.append(label[0])
                class_weights.update(label[0])
            logger.info("creating sampler totally: %d", len(self.data_list))

            if len(class_weights) != self.num_classes:
                logger.warning("{} classes have no samples.".format(self.num_classes-len(class_weights)))
                for cls in range(self.num_classes):
                    if cls not in class_weights:
                        class_weights.update({cls:1e9})

            sorted_weights = sorted(class_weights.items(), key=lambda x: x[0])
            weights = [s[1] for s in sorted_weights]
            weights = 1.0 / np.array(weights)
            sample_weights = []
            for i, sample_label in enumerate(sample_labels):
                if i % 100000 == 0:
                    logger.info("prcessoing sample weight: %d/%d", i, len(sample_labels))
                ws = weights[sample_label]
                w = float(np.mean(ws))
                sample_weights.append(w)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples,
                                            replacement=replacement)
            del weights
            del class_weights
            del sorted_weights
            del sample_weights
            del sample_labels
            return sampler
        else:
            return None
