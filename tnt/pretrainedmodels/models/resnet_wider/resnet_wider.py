import torch
import torch.nn as nn


model_names = ["resnet101x1", "resnet101x1_sk"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SK_Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, stride, sk_ratio, min_dim=32):
        super(SK_Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes*2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.mid_dim = max(int(out_planes * sk_ratio), min_dim)
        self.conv2 = nn.Conv2d(out_planes, self.mid_dim, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_dim)
        self.conv3 = nn.Conv2d(self.mid_dim, out_planes*2, kernel_size=1, stride=1, bias=False)
        self.out_planes = out_planes

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        half = x.shape[1] // 2
        x = torch.stack(torch.split(x, [half, half], dim=1))
        global_features = torch.mean(torch.sum(x, dim=0), dim=[2, 3], keepdim=True)
        global_features = self.conv2(global_features)
        global_features = self.bn2(global_features)
        global_features = self.relu(global_features)
        mixing = self.conv3(global_features)
        mixing = torch.stack(torch.split(mixing, [self.out_planes, self.out_planes], dim=1))
        mixing = torch.softmax(mixing, dim=0)

        output = torch.sum(x*mixing, dim=0)
        return output


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, sk_ratio=0):
        super(Bottleneck, self).__init__()
        self.downsample = downsample  # hack: moving downsample to the first to make order correct
        self.sk_ratio = sk_ratio
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.sk_ratio > 0:
            self.sk_conv2d = SK_Conv2d(width, width, stride, sk_ratio)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.sk_ratio > 0:
            out = self.sk_conv2d(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, width_mult=1, sk_ratio=0.0, use_head=2):
        super(ResNet, self).__init__()
        self.input_space = "RGB"
        self.input_range = [0, 1]
        self.input_size = [3, 224, 224]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 * width_mult
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.sk_ratio = sk_ratio
        if self.sk_ratio > 0:
            self.conv1 = nn.Conv2d(3, self.inplanes//2, kernel_size=3, stride=2, padding=1)
            self.conv_bn1 = norm_layer(self.inplanes//2)
            # add relu
            self.conv2 = nn.Conv2d(self.inplanes//2, self.inplanes//2, kernel_size=3, stride=1, padding=1)
            self.conv_bn2 = norm_layer(self.inplanes//2)
            # add relu
            self.conv3 = nn.Conv2d(self.inplanes//2, self.inplanes, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * width_mult, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256 * width_mult, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512 * width_mult, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        head_dim = 512 * block.expansion * width_mult
        self.use_head = use_head
        if self.use_head >= 1:
            self.head1 = nn.Linear(head_dim, head_dim, bias=False)
            self.head_bn1 = nn.BatchNorm1d(head_dim)
        if self.use_head >= 2:
            self.head2 = nn.Linear(head_dim, head_dim, bias=False)
            self.head_bn2 = nn.BatchNorm1d(head_dim)
        self.fc = nn.Linear(head_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.sk_ratio > 0:
                downsample = nn.Sequential(
                    nn.ReplicationPad2d(padding=(0, 1, 0, 1)),
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.sk_ratio))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, sk_ratio=self.sk_ratio))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if self.sk_ratio > 0:
            x = self.conv1(x)
            x = self.conv_bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.conv_bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_head >= 1:
            x = self.head1(x)
            x = self.head_bn1(x)
            x = self.relu(x)
        if self.use_head >= 2:
            x = self.head2(x)
            x = self.head_bn2(x)
            x = self.relu(x)

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet101x1_sk(**kwargs):
    use_head = kwargs.get("use_head", 2)
    num_classes = kwargs.get("num_classes", 1000)
    pretrained = kwargs.get("pretrained", None)
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_mult=1, sk_ratio=0.0625, use_head=use_head)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        missing_keys, unexcepted_keys = model.load_state_dict(state_dict, strict=False)
        print("loading pretrained weights, missing_keys:{}, unexcepted_keys:{}".format(
            missing_keys, unexcepted_keys))
    return model


def resnet101x1(**kwargs):
    use_head = kwargs.get("use_head", 2)
    num_classes = kwargs.get("num_classes", 1000)
    pretrained = kwargs.get("pretrained", None)
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_mult=1, sk_ratio=0, use_head=use_head)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        missing_keys, unexcepted_keys = model.load_state_dict(state_dict, strict=False)
        print("loading pretrained weights, missing_keys:{}, unexcepted_keys:{}".format(
            missing_keys, unexcepted_keys))
    return model


if __name__ == "__main__":
    sk_conv2d = SK_Conv2d(64, 128, 2, 0.0625, 32)
    x = torch.rand(10, 64, 14, 14)
    output = sk_conv2d(x)
    print("output:", output.shape)

    ##
    """
    model = resnet101x1(use_head=2, num_classes=1000)
    model.eval()
    input = torch.rand(10, 3, 224, 224)
    output = model(input)
    """

    model = resnet101x1_sk(use_head=2, num_classes=1000)
    model.eval()
    input = torch.rand(10, 3, 224, 224)
    output = model(input)
    print("ok.")
