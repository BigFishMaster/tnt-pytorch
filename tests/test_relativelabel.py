import torch
import torch.nn.functional as F

batch_size = 5
num_classes = 6
torch.manual_seed(123)
data = torch.rand(batch_size, num_classes)
index = [
    [1, 3, 4],
    [4, 5, 1],
    [2, 4,-1],
    [1,-1,-1],
    [3,-1,-1],
]
index = torch.tensor(index)
print("1 ok.")
print("data:", data.size())
print("index:", index.size())

# start loss:
pos_mask = index != -1
sample_mask = pos_mask.float().sum(1) > 1
# positive predictions
neg_mask = index == -1
index[neg_mask] = 0
pos_data = torch.gather(data, dim=1, index=index)
pos_data[neg_mask] = 1e8
min_relative_index = pos_data.argmin(1, keepdim=True)
min_relative_data = torch.gather(pos_data, dim=1, index=min_relative_index)
# negative predictions
pos_mask = pos_mask.view(-1)
bias = torch.arange(batch_size, dtype=torch.long, device=data.device) * num_classes
new_index = bias.view(batch_size, 1) + index
new_index = new_index.view(-1)
# available positive mask
new_index = new_index[pos_mask]
new_data = data.contiguous().view(-1)
new_data[new_index] = -1e8
new_data = new_data.view(batch_size, num_classes)
relative_data = torch.cat([min_relative_data, new_data], dim=1)
relative_data = relative_data[sample_mask]
target_label = torch.zeros(relative_data.size(0), dtype=torch.long, device=relative_data.device)
loss2 = F.cross_entropy(relative_data, target_label)
print("loss2:", loss2)

