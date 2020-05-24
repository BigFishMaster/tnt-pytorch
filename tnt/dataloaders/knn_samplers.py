import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F
from tnt.utils.logging import logger
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader
from munch import munchify
from tnt.data.transform_image import TransformImage
from tnt.utils.io import load_image_from_path


class KNNDataLoader(Dataset):
    def __init__(self, dic1, batch_size, data_prefix):
        data_list = []
        num_labels = len(dic1)
        for label in range(num_labels):
            samples = dic1[label]
            samples = samples[:batch_size]
            data_list.extend(samples)
        self.data_list = data_list
        logger.info("In DataLoader, data length:{}".format(len(self.data_list)))

        opts = {"input_size": [3, 224, 224],
                "input_space": "RGB",
                "input_range": [0, 1],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "five_crop": False,
                "ten_crop": False,
                "preserve_aspect_ratio": 0,
                "random_erase": False,
                "image_scale": 0.875,
                "random_crop": False,
                "random_hflip": False,
                "random_vflip": False,
                }
        opts = munchify(opts)
        #TODO: data_prefix setting.
        self.data_prefix = data_prefix
        self.transform = TransformImage(opts)

    @classmethod
    def from_dict(cls, dic1, batch_size, data_prefix):
        self = cls(dic1, batch_size, data_prefix)
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return data_loader

    def __getitem__(self, index):
        data = self.data_list[index]
        path, label = data.split()
        label = int(label)
        image = load_image_from_path(path, data_prefix=self.data_prefix,
                                     transforms=self.transform)
        return image, label

    def __len__(self):
        return len(self.data_list)


class KNNSampler(Sampler):
    def __init__(self, label2index, each_class, num_samples, filename, data_prefix):
        self.num_labels = len(label2index)
        self.label2index = label2index
        # sampling: totally num_samples and each_class for one label.
        self.each_class = each_class
        self.num_samples = num_samples
        self.sampling_depth = 2
        self.target_num = num_samples // each_class
        # TODO: include itself.
        self.knn = None
        self.knn_num = 8
        self.dim = 512
        self.update_steps = 10
        self.steps = 0
        self.features = torch.zeros(self.num_labels, self.dim)
        self.losses = torch.zeros(self.num_labels)

        data = open(filename, "r", encoding="utf8").readlines()
        dic1 = {}
        for i, item in enumerate(data):
            path, label = item.strip().split()
            label = int(label)
            if label not in dic1:
                dic1[label] = []
            dic1[label].append(item.strip())
        logger.info("labels to initialize knn: {}".format(len(dic1)))
        self.dataloader = KNNDataLoader.from_dict(dic1, batch_size=5, data_prefix=data_prefix)

    def build(self, model):
        model.eval()
        with torch.no_grad():
            for label, batch in enumerate(self.dataloader):
                input, target = batch
                output = model(input)
                feature = output.mean(0)
                self.features[label] = feature
                output_norm = F.normalize(output)
                loss = 1.0 - torch.matmul(output_norm, output_norm.t()).min().item()
                self.losses[label] = max(loss, 1e-6)

            self.features = F.normalize(self.features)
            distance = torch.matmul(self.features, self.features.t())
            _, self.knn = distance.topk(self.knn_num, 1, True, True)
            self.knn = self.knn.tolist()
            logger.info("initialize knn: {}".format(self.knn))

    def update(self, features, labels):
        with torch.no_grad():
            batch_size = len(labels)
            num_label = batch_size // self.each_class
            for i in range(num_label):
                start = i * self.each_class
                end = start + self.each_class
                local_features = features[start:end]
                local_features_norm = F.normalize(local_features)
                loss = torch.matmul(local_features_norm, local_features_norm.t()).min().item()
                feature = F.normalize(local_features.mean(0), dim=0)
                label = labels[start]
                self.features[label] = feature
                self.losses[label] = max(1 - loss, 1e-6)

            self.steps = (self.steps + 1) % self.update_steps
            if self.steps == 0:
                distance = torch.matmul(self.features, self.features.t())
                _, self.knn = distance.topk(self.knn_num, 1, True, True)
                self.knn = self.knn.tolist()
                s, p = self.losses.topk(10, 0, True, True)
                logger.info("KNN Tree is updated. "
                            "top-10 loss with score: {}. label: {}.".format(s, p))

    def _select(self, label, output, depth=0):
        if depth >= self.sampling_depth:
            return
        knn_labels = [l for l in self.knn[label] if l != label and l not in output]
        selected_length = min(self.target_num-len(output), len(knn_labels))
        selected_labels = knn_labels[:selected_length]
        output.update(selected_labels)
        for lab in selected_labels:
            if len(output) >= self.target_num:
                break
            self._select(lab, output, depth+1)

    def _sample(self):
        cand_labels = torch.multinomial(self.losses, self.target_num).tolist()
        output = Counter()
        for label in cand_labels:
            if len(output) >= self.target_num:
                break
            if label not in output:
                output.update([label])
            self._select(label, output, 0)
        selected_labels = list(output.keys())
        return selected_labels

    def __iter__(self):
        selected_labels = self._sample()
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
