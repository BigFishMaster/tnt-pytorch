import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from tnt.utils.logging import logger
from tnt.dataloaders.field import Field


class GeneralDataLoader(Dataset):
    def __init__(self, cfg, mode="train"):
        self.num_classes = cfg["num_classes"]
        filename = cfg[mode]
        self.data_list = open(filename, "r", encoding="utf8").readlines()
        logger.info("In mode {}, data_list has length of {}.".format(mode, len(self.data_list)))

        # field processor
        self._field = Field.from_cfg(cfg)

        # sampling strategy
        sampler_config = cfg["sampler"]
        self.sampler = None
        if mode == "train" and sampler_config is not None:
            self.sampler = self._create_sampler(sampler_config)
        else:
            self.sampler = None
        self.batch_size = sampler_config.get("batch_size", 10)
        self.num_workers = sampler_config.get("num_workers", 4)

    @classmethod
    def from_config(cls, cfg, mode="train"):
        if (mode not in cfg) or (not cfg[mode]):
            return None
        self = cls(cfg, mode)
        data_loader = DataLoader(self, batch_size=self.batch_size, shuffle=(self.sampler is None),
                                 num_workers=self.num_workers, pin_memory=True, sampler=self.sampler)
        return data_loader

    def __getitem__(self, index):
        data = self.data_list[index]
        return self._field(data)

    def __len__(self):
        return len(self.data_list)

    def _create_sampler(self, cfg):
        strategy = cfg.get("strategy")
        num_samples = cfg.get("num_samples") or len(self.data_list)
        replacement = cfg.get("replacement") or True
        sample_labels = []
        if strategy == "class_balanced":
            for i, data in enumerate(self.data_list):
                if i % 10000 == 0:
                    logger.info("creating sampler for data: %d/%d", i, len(self.data_list))
                label = self._field(data, last=True)
                sample_labels.append(label[0])
            logger.info("creating sampler totally: %d", len(self.data_list))

        class_weights = Counter()
        class_weights.update(sample_labels)
        if len(class_weights) != self.num_classes:
            raise ValueError("{} classes have no samples.".format(self.num_classes-len(class_weights)))

        sorted_weights = sorted(class_weights.items(), key=lambda x: x[0])
        weights = [s[1] for s in sorted_weights]
        weights = 1.0 / np.array(weights)
        sample_weights = weights[sample_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples,
                                        replacement=replacement)
        return sampler

