import torch
torch.manual_seed(111)
feature = torch.rand((10, 16, 512)).double()
weight = torch.rand(16, 512, 100).double()

f1 = feature.unsqueeze(3)
w1 = weight.unsqueeze(0)
o1 = f1 * w1
o1 = o1.sum(dim=2)
print(o1.shape)

o2 = []
for i in range(16):
    f0 = feature[:, i, :]
    w0 = weight[i, :, :]
    o0 = torch.matmul(f0, w0).unsqueeze(1)
    o2.append(o0)
o2 = torch.cat(o2, dim=1)
print(o2.shape)

B, N, D = o2.shape
diff = o1 - o2
print("diff:", diff, diff.max(), diff.min())

