import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F
from tnt.utils.logging import logger, beautify_info
from torch.utils.data import Sampler
from torch.utils.data import Dataset, DataLoader
from munch import munchify
from tnt.data.transform_image import TransformImage
from tnt.utils.io import load_image_from_path


class KNNDataLoader(Dataset):
    def __init__(self, dic1, num_each_class, data_prefix):
        data_list = []
        num_labels = len(dic1)
        for label in range(num_labels):
            samples = dic1[label]
            if len(samples) < num_each_class:
                samples = samples * num_each_class
                samples = samples[:num_each_class]
            else:
                samples = samples[:num_each_class]
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
                "erase_count": None,
                "random_erase": False,
                "image_scale": 1.0,
                "random_crop": False,
                "random_hflip": False,
                "random_vflip": False,
                }
        opts = munchify(opts)
        #TODO: data_prefix setting.
        self.data_prefix = data_prefix
        self.transform = TransformImage(opts)

    @classmethod
    def from_dict(cls, dic1, num_each_class, batch_size, data_prefix):
        self = cls(dic1, num_each_class, data_prefix)
        data_loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
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
    def __init__(self, label2index, batch_size, each_class, num_samples, filename, data_prefix):
        self.num_labels = len(label2index)
        self.label2index = label2index
        # sampling: totally num_samples and each_class for one label.
        if num_samples % batch_size != 0:
            raise ValueError("In KNNSampler, num_samples {} must be divided by batch_size {}.".format(
                num_samples, batch_size))
        self.each_class = each_class
        self.num_samples = num_samples
        self.target_num = num_samples // each_class
        self.target_each_batch = batch_size // each_class
        # duplicate the depth
        self.sampling_depth = self.target_each_batch * 2
        logger.info("num_samples: {}, each_class: {}, target_num: {}, target_each_class:{}".format(
            self.num_samples, self.each_class, self.target_num, self.target_each_batch))
        # TODO: include itself.
        self.batch_size = batch_size
        self.knn = None
        self.knn_num = 8
        self.dim = 512
        self.update_steps = 1000
        self.steps = 0
        self.print_steps = self.update_steps
        self.p_steps = 0
        torch.manual_seed(0)
        self.features = torch.rand(self.num_labels, self.dim)
        self.losses = torch.rand(self.num_labels)
        self.knn = torch.randint(0, self.num_labels, (self.num_labels, self.knn_num)).tolist()

        data = open(filename, "r", encoding="utf8").readlines()
        dic1 = {}
        for _, (label, v) in enumerate(label2index.items()):
            dic1[label] = [data[i].strip() for i in v]
        logger.info("labels to initialize knn: {}".format(len(dic1)))
        # Note: each batch will contain multiple labels, like: 0 0 0 0 1 1 1 1 2 2 2 2
        if self.batch_size % self.each_class != 0:
            raise ValueError("When build knn trees, batch_size {} must be divided by num_each_class {}.".format(
                self.batch_size, self.each_class))
        self.dataloader = KNNDataLoader.from_dict(dic1, num_each_class=self.each_class, batch_size=self.batch_size,
                                                  data_prefix=data_prefix)

    def build(self, model):
        model.eval()
        with torch.no_grad():
            for label, batch in enumerate(self.dataloader):
                if label % 100 == 0:
                    logger.info("building knn sampler: {} labels".format(label*self.batch_size//self.each_class))
                input, target = batch
                output = model(input)
                output = output.reshape(-1, self.each_class, self.dim)
                cur_label_num = len(output)
                for i in range(cur_label_num):
                    # num_each_class x dim
                    output_norm = F.normalize(output[i])
                    one_label = int(target[i*self.each_class].item())
                    self.features[one_label] = output_norm.mean(0)
                    loss = 1.0 - torch.matmul(output_norm, output_norm.t()).min().item()
                    self.losses[one_label] = max(loss, 1e-6)

            distance = torch.matmul(self.features, self.features.t())
            _, self.knn = distance.topk(self.knn_num, 1, True, True)
            self.knn = self.knn.tolist()
            logger.info("initialize knn: {}".format(beautify_info(self.knn[:10])))

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

            self.p_steps = (self.p_steps + 1) % self.print_steps
            if self.p_steps == 0:
                s, p = self.losses.topk(10, 0, True, True)
                logger.info("Hard Sampling with top-k label: {}.".format(p))
                logger.info("Hard Sampling with top-k score: {}.".format(s))

    def _select_bfs(self, label, output, depth=0):
        if depth >= self.sampling_depth:
            return
        knn_labels = [l for l in self.knn[label] if l != label and l not in output]
        selected_length = min(self.target_each_batch-len(output), len(knn_labels))
        selected_labels = knn_labels[:selected_length]
        output.update(selected_labels)
        for lab in selected_labels:
            if len(output) >= self.target_each_batch:
                break
            self._select_bfs(lab, output, depth+1)

    def _select_dfs(self, label, output, depth=0):
        if depth >= self.sampling_depth:
            return
        selected_label = None
        for l in self.knn[label]:
            if l != label and l not in output:
                selected_label = l
                break
        if selected_label is None:
            return

        output.update([selected_label])
        if len(output) >= self.target_each_batch:
            return
        else:
            self._select_dfs(selected_label, output, depth+1)

    def _sample(self):
        selected_labels = []
        loops = self.target_num // self.target_each_batch
        for i in range(loops):
            cand_labels = torch.multinomial(self.losses, self.target_each_batch).tolist()
            output = Counter()
            for label in cand_labels:
                if len(output) >= self.target_each_batch:
                    break
                if label not in output:
                    output.update([label])
                self._select_dfs(label, output, 0)
            batch_labels = list(output.keys())
            selected_labels.extend(batch_labels[:self.target_each_batch])
        if len(selected_labels) != self.target_num:
            raise ValueError("selected_labels {} is not equal to target_num {}.".format(
                selected_labels, self.target_num))
        return selected_labels

    def __iter__(self):
        selected_labels = self._sample()
        if self.p_steps == 0:
            logger.info("selected labels: {}".format(beautify_info(selected_labels[:self.target_each_batch])))
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
