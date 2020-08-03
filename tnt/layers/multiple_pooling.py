import torch
import torch.nn as nn


class MultiplePooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiplePooling, self).__init__()
        # output: B x C x 1 x 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.way_0 = self._make_layer(input_dim, output_dim)
        self.way_1 = self._make_layer(input_dim, output_dim)
        self.way_2 = self._make_layer(input_dim, output_dim)
        self.way_3 = self._make_layer(input_dim, output_dim)
        self.way_4 = self._make_layer(input_dim, output_dim)
        self.way_5 = self._make_layer(input_dim, output_dim)
        self.way_6 = self._make_layer(input_dim, output_dim)
        self.way_7 = self._make_layer(input_dim, output_dim)
        self.way_8 = self._make_layer(input_dim, output_dim)
        self.way_9 = self._make_layer(input_dim, output_dim)
        self.way_10 = self._make_layer(input_dim, output_dim)
        self.way_11 = self._make_layer(input_dim, output_dim)
        self.way_12 = self._make_layer(input_dim, output_dim)
        self.way_13 = self._make_layer(input_dim, output_dim)
        self.way_14 = self._make_layer(input_dim, output_dim)
        self.way_15 = self._make_layer(input_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            #nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # input: B x C x 12 x 12
        h, w = input.shape[2:4]
        p0 = self.gap(input)
        step0 = h // 2
        # B x C x 6 x 12
        h2_1, h2_2 = torch.split(input, (step0, h-step0), dim=2)
        p1 = self.gap(h2_1)
        p2 = self.gap(h2_2)
        step1 = h // 3
        # B x C x 4 x 12
        h3_1, h3_2, h3_3 = torch.split(input, (step1, step1, h-2*step1), dim=2)
        p3 = self.gap(h3_1)
        p4 = self.gap(h3_2)
        p5 = self.gap(h3_3)
        # B x C x 12 x 6
        w2_1, w2_2 = torch.split(input, (step0, w-step0), dim=3)
        p6 = self.gap(w2_1)
        p7 = self.gap(w2_2)
        w3_1, w3_2, w3_3 = torch.split(input, (step1, step1, w-2*step1), dim=3)
        p8 = self.gap(w3_1)
        p9 = self.gap(w3_2)
        p10 = self.gap(w3_3)
        # annular 2
        # 6 x 6
        a2_s = h // 4
        a2_e = h * 3 // 4
        a2 = input[:, :, a2_s:a2_e, a2_s:a2_e]
        a2_h, a2_w = a2.shape[2:4]
        p11 = self.gap(a2)
        area0 = h * w - a2_h * a2_w
        p12 = (p0 * h * w - p11 * a2_h * a2_w) / area0
        # annular 3
        # 8 x 8
        a3_s1 = h // 6
        a3_e1 = h * 5 // 6
        # 4 x 4
        a3_s2 = h // 3
        a3_e2 = h * 2 // 3
        a3_1 = input[:, :, a3_s1:a3_e1, a3_s1:a3_e1]
        a3_2 = input[:, :, a3_s2:a3_e2, a3_s2:a3_e2]
        # B x C x 8 x 8
        a3_1_h, a3_1_w = a3_1.shape[2:4]
        # B x C x 4 x 4
        a3_2_h, a3_2_w = a3_2.shape[2:4]
        p13 = self.gap(a3_2)
        a3_8x8 = self.gap(a3_1)
        area1 = h * w - a3_1_h * a3_1_w
        area2 = a3_1_h * a3_1_w - a3_2_h * a3_2_w
        p14 = (p0 * h * w - a3_8x8 * a3_1_h * a3_1_w) / area1
        p15 = (a3_8x8 * a3_1_h * a3_1_w - p13 * a3_2_h * a3_2_w) / area2

        fea0 = self.way_0(p0)
        fea1 = self.way_1(p1)
        fea2 = self.way_2(p2)
        fea3 = self.way_3(p3)
        fea4 = self.way_4(p4)
        fea5 = self.way_5(p5)
        fea6 = self.way_6(p6)
        fea7 = self.way_7(p7)
        fea8 = self.way_8(p8)
        fea9 = self.way_9(p9)
        fea10 = self.way_10(p10)
        fea11 = self.way_11(p11)
        fea12 = self.way_12(p12)
        fea13 = self.way_13(p13)
        fea14 = self.way_14(p14)
        fea15 = self.way_15(p15)

        # shape: B x C*16 x 1 x 1
        output = torch.cat([fea0, fea1, fea2, fea3, fea4, fea5, fea6,
                            fea7, fea8, fea9, fea10, fea11, fea12, fea13,
                            fea14, fea15], dim=1)
        output = torch.flatten(output, 1)
        return output


if __name__ == "__main__":
    input = torch.Tensor(5, 32, 12, 12)
    layer = MultiplePooling(32, 16)
    layer.eval()
    output = layer(input)

