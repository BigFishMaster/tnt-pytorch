import torch
import torch.nn.functional as F


class MultiLabelLoss(torch.nn.Module):

    def __init__(self):
        super(MultiLabelLoss, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: batch_size x class_dim
            y: batch_size x target_dim; please ignore -1.

        Returns:

        """
        #x = torch.rand(10, 100)
        #y = torch.rand(10, 4)
        logsoftmax = F.log_softmax(x, 1)
        batch_size, class_dim = x.shape
        _, target_dim = y.shape
        loss = 0
        for i in range(batch_size):
            data = logsoftmax[i]
            index = y[i]
            flag = index.ne(-1)
            index_selected = (index[flag]).long()
            pred = torch.gather(data, 0, index_selected)
            scale = 1 / pred.numel()
            loss0 = -torch.sum(pred) * scale
            loss += loss0
        loss = loss / batch_size
        return loss


def test():
    loss_fn = MultiLabelLoss()
    x = torch.rand(4, 10)
    y = torch.Tensor([[1,2,0,-1,-1], [1,-1,-1,-1,-1],[2,0,-1,-1,-1],[3,2,1,0,4]])
    loss = loss_fn(x, y)
    print(loss)


if __name__ == "__main__":
    test()
