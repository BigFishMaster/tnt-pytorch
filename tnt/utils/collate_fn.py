import torch
import numpy as np


def multilabel_collate_fn(batch):
    max_len = max(map(lambda x: len(x[1]), batch))
    img, label = zip(*batch)
    pad_label = []
    for i in range(len(label)):
        temp_label = np.array([-1] * max_len)
        temp_label[:len(label[i])] = label[i]
        pad_label.append(torch.from_numpy(temp_label))
    img = torch.stack(img, 0)
    pad_label = torch.stack(pad_label, 0).long()
    return img, pad_label