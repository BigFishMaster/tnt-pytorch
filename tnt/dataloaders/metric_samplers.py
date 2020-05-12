import random
from torch.utils.data import Sampler
import numpy as np


class MetricDataSampler(Sampler):

    def __init__(self, label2index, each_class, num_samples):
        if num_samples % each_class != 0:
            raise ValueError("num_samples {} can not be divided by each_class {}".format(num_samples, each_class))
        self.label2index = label2index
        self.each_class = each_class
        self.num_samples = num_samples
        self.target_num = self.num_samples // self.each_class
        self.label_list = list(self.label2index.keys())
        if self.target_num > len(self.label_list):
            raise ValueError("sampling labels {} can not be larger than total labels {}.".format(
                self.target_num, self.label_list))

    def __iter__(self):
        random.shuffle(self.label_list)
        selected_labels = self.label_list[:self.target_num]
        sample_list = []
        for sl in selected_labels:
            indices = self.label2index[sl]
            if len(indices) > self.each_class:
                selected_index = np.random.choice(indices, size=self.each_class, replace=False)
            else:
                selected_index = np.random.choice(indices, size=self.each_class, replace=True)
            sample_list.extend(list(selected_index))

        return iter(sample_list)

    def __len__(self):
        return self.num_samples
