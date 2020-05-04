import torch
import torch.nn.functional as F


class RelativeLabelLoss(torch.nn.Module):

    def __init__(self, gamma=0.2):
        super(RelativeLabelLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.gamma = gamma

    def forward(self, x, y):
        """
        Args:
            x: batch_size x class_dim
            y: batch_size x target_dim

        Returns:

        """
        y = y.long()
        target = y[:, 0]
        loss1 = self.ce(x, target)
        batch_size, class_dim = x.shape
        count = 1e-8
        loss2 = 0
        for i in range(batch_size):
            data = x[i]
            index = y[i][1:]
            flag = index.ne(-1)
            index_selected = index[flag]
            if len(index_selected) == 0:
                continue
            count += 1
            ones = torch.ones(class_dim)
            ones[index_selected] = 0
            pred = data.gather(0, index_selected)
            min_pred_index = pred.argmin()
            relative_index = index_selected[min_pred_index]
            ones[relative_index] = 1
            cand_index = ones.nonzero().view(-1)
            cand_data = torch.gather(data, 0, cand_index)
            relative_loss = F.cross_entropy(cand_data.view(1,-1), relative_index.view(-1))
            loss2 += relative_loss

        loss = loss1 + self.gamma * loss2/count
        return loss


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


class WeightLabelLoss(torch.nn.Module):

    def __init__(self):
        super(WeightLabelLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y):
        """
        Args:
            x: batch_size x class_dim
            y: a list with [target, score],
                 target: batch_size
                 score: batch_size
        Returns:

        """
        #x = torch.rand(10, 100)
        #y = [torch.randint(0, 100, (10,)), torch.rand(10,)]
        w, t = y
        l = self.ce(x, t)
        w_l = w.float() * l
        w_l = w_l.mean()
        return w_l


def test_relativelabelloss():
    loss_fn = RelativeLabelLoss()
    x = torch.rand(4, 10)
    y = torch.Tensor([[1,2,0,-1,-1], [1,-1,-1,-1,-1],[2,-1,-1,-1,-1],[3,2,1,0,4]])
    loss = loss_fn(x, y)
    print(loss)


def test_multilabelloss():
    loss_fn = MultiLabelLoss()
    x = torch.rand(4, 10)
    y = torch.Tensor([[1,2,0,-1,-1], [1,-1,-1,-1,-1],[2,0,-1,-1,-1],[3,2,1,0,4]])
    loss = loss_fn(x, y)
    print(loss)


def test_weightlabelloss():
    loss_fn = WeightLabelLoss()
    x = torch.rand(4, 10)
    y = [torch.Tensor([2,6,4,8]), torch.Tensor([0.3,0.2,0.4,0.7])]
    loss = loss_fn(x, y)
    print(loss)


if __name__ == "__main__":
    #test_multilabelloss()
    #test_weightlabelloss()
    test_relativelabelloss()
