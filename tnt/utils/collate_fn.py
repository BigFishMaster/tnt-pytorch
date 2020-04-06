import torch
import numpy as np

def multilabel_collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = np.array([-1] * max_len)
        temp_label[:len(label[i])] = label[i]
        pad_label.append(torch.from_numpy(temp_label))
    img = torch.stack(img, 0)
    pad_label = torch.stack(pad_label, 0)
    return img, pad_label