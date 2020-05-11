"""
Pytorch implementation of Class-Balanced-Loss
Reference: "Class-Balanced Loss Based on Effective Number of Samples"
Authors: Yin Cui and Menglin Jia and Tsung Yi Lin and Yang Song and Serge J. Belongie
https://arxiv.org/abs/1901.05555, CVPR'19.
github: https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
"""
import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss


class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_class=None, num_classes=5000, beta=0.9999, gamma=0.5, loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        if loss_type not in ["focal", "sigmoid", "softmax"]:
            loss_type = "focal"
        if samples_per_class is None:
            samples_per_class = [1] * num_classes
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        total_sum = 10.0
        weights = (weights / np.sum(weights) * total_sum).astype(np.float32)

        self.class_weights = weights
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

    def update(self, samples_per_class):
        if samples_per_class is None:
            return
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = (weights / np.sum(weights) * self.num_classes).astype(np.float32)

        self.class_weights = weights

    def forward(self, x, y):
        labels_one_hot = F.one_hot(y, self.num_classes).float()
        weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y)
        weights = weights.unsqueeze(1)
        if self.loss_type == "focal":
            cb_loss = focal_loss(x, labels_one_hot, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(x, labels_one_hot, weights)
        else:  # softmax
            pred = x.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights)
        return cb_loss


def test():
    torch.manual_seed(123)
    batch_size = 10
    num_classes = 5
    x = torch.rand(batch_size, num_classes)
    y = torch.randint(0, 5, size=(batch_size,))
    samples_per_class = [1, 2, 3, 4, 5]
    loss_type = "focal"
    loss_fn = ClassBalancedLoss(samples_per_class, num_classes, loss_type=loss_type)
    loss = loss_fn(x, y)
    print(loss)


if __name__ == '__main__':
    test()
