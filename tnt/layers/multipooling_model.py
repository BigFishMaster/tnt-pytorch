import torch
import torch.nn as nn
import tnt.pretrainedmodels as pretrainedmodels
from tnt.layers.multiple_pooling import MultiplePooling


class MultiPoolingModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained):
        super(MultiPoolingModel, self).__init__()
        self.base = pretrainedmodels.__dict__[model_name](pretrained=pretrained, multiple_pooling=True)
        self.mp = MultiplePooling(input_dim=2048, output_dim=num_classes)
        self.input_space = self.base.input_space
        self.input_range = self.base.input_range
        self.input_size = self.bsae.input_size
        self.mean = self.base.mean
        self.std = self.base.std

    def forward(self, x):
        # shape: B x C_in x 12 x 12
        conv_feature = self.base(x)
        # shape: B x C_out*16
        out_feature = self.mp(conv_feature)
        return out_feature


if __name__ == "__main__":
    mp = MultiPoolingModel("resnest101", num_classes=128, pretrained=None)
    mp.eval()
    x = torch.rand((2, 3, 384, 384))
    label = torch.randint(0, 10, size=(2,))
    from tnt.losses.face_loss import CosFaceLoss
    loss_fn = CosFaceLoss(feature_size=128, num_classes=10)
    out = mp(x)
    if label is not None and len(out) > len(label):
        num = len(out) // len(label)
        label = label.repeat(num)
    loss = loss_fn(out, label)
    print("loss:", loss)




