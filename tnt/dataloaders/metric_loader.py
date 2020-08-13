import numpy as np
from torch.utils.data import Dataset, DataLoader
from tnt.utils.logging import logger, beautify_info
from tnt.dataloaders.field import Field
from tnt.dataloaders.metric_samplers import MetricDataSampler
from tnt.dataloaders.knn_samplers import KNNSampler


class MetricDataLoader(Dataset):
    def __init__(self, cfg, mode="train"):
        self.num_classes = cfg["num_classes"]
        filename = cfg[mode]
        self.data_list = open(filename, "r", encoding="utf8").readlines()
        self.data_list = np.array(self.data_list, dtype=np.string_)
        logger.info("In mode {}, data_list has length of {}.".format(mode, len(self.data_list)))


        # for hard sampling
        self.filename = filename
        self.data_prefix = cfg["data_prefix"]
        # field processor
        self._field = Field.from_cfg(cfg, mode=mode)

        # batch sampling strategy
        sampler_config = cfg["sampler"]
        self.sampler = self._create_sampler(sampler_config, mode)

        self.batch_size = sampler_config.get("batch_size", 50)
        self.num_workers = sampler_config.get("num_workers", 4)

    @classmethod
    def from_config(cls, cfg, mode="train"):
        if (mode not in cfg) or (not cfg[mode]):
            return None
        self = cls(cfg, mode)
        pin_memory = cfg["pin_memory"]
        data_loader = DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers,
                                 pin_memory=pin_memory, sampler=self.sampler, drop_last=(mode != "test"))
        logger.info("data loader is: {}".format(beautify_info(data_loader)))
        return data_loader

    def __getitem__(self, index):
        data = self.data_list[index].decode()
        result = self._field(data)
        return result

    def __len__(self):
        return len(self.data_list)

    def _create_sampler(self, cfg, mode):
        if mode == "test":
            return None
        label2index = {}
        for i, data in enumerate(self.data_list):
            if i % 10000 == 0:
                logger.info("creating sampler for data: %d/%d", i, len(self.data_list))
            label = self._field(data.decode(), last=True)
            label = label[0]
            if label not in label2index:
                label2index[label] = []
            label2index[label].append(i)

        logger.info("In mode {}, the length of label2index is {}.".format(mode, len(label2index)))
        each_class = cfg.get("each_class") or 5
        if mode == "train":
            num_samples = cfg.get("num_samples") or (len(label2index) * each_class)
            batch_size = cfg.get("batch_size") or (20 * each_class)
        else:  # valid
            num_samples = len(label2index) * each_class
            batch_size = 20 * each_class
        if batch_size % each_class != 0:
            raise ValueError("batch_size {} can not be divided by each_class {}.".format(batch_size, each_class))

        strategy = cfg.get("strategy")
        if strategy == "knn_sampler" and mode == "train":
            return KNNSampler(label2index, batch_size, each_class, num_samples, self.filename, self.data_prefix)
        else:
            return MetricDataSampler(label2index, each_class, num_samples)
