import numpy as np


def generate_index(batch_size, samples_each_class):
    a = np.arange(0, batch_size * batch_size) # N*N x 1
    a = a.reshape(-1, batch_size) # N x N
    steps = batch_size // samples_each_class
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
    res = np.array(res).astype(np.int32)
    return res


