import numpy as np
import torch

"""
If batch_size=9, samples_each_class=3,
return a LongTensor with shape: torch.Size([81])
LongTensor:
tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10,  9, 11, 12, 13, 14, 15, 16, 17,
        20, 18, 19, 21, 22, 23, 24, 25, 26, 30, 31, 32, 27, 28, 29, 33, 34, 35,
        40, 39, 41, 36, 37, 38, 42, 43, 44, 50, 48, 49, 45, 46, 47, 51, 52, 53,
        60, 61, 62, 54, 55, 56, 57, 58, 59, 70, 69, 71, 63, 64, 65, 66, 67, 68,
        80, 78, 79, 72, 73, 74, 75, 76, 77])

"""


def generate_index(batch_size, samples_each_class):
    a = np.arange(0, batch_size * batch_size)  # N*N x 1
    a = a.reshape(-1, batch_size)  # N x N
    res = []
    for i in range(batch_size):
        step = i // samples_each_class
        start = step * samples_each_class
        end = (step + 1) * samples_each_class
        p = []
        n = []
        for j, k in enumerate(a[i]):
            if j >= start and j < end:
                if j == i:
                    p.insert(0, k)
                else:
                    p.append(k)
            else:
                n.append(k)
        comb = p + n
        res += comb
    res = torch.from_numpy(np.array(res)).long()
    return res
