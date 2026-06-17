import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math

def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = t.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

def conv_1x1_bn(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2

        if downsample:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(inplace=True)
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.downsample:
            out = t.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            c = x.shape[1] // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = t.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)

class ShuffleNet2Fer(nn.Module):
    def __init__(self, num_classes=7, input_size=48, net_type=1):
        super(ShuffleNet2Fer, self).__init__()

        self.stage_repeat_num = [4, 8, 4]
        if net_type == 0.5:
            self.out_channels = [1, 24, 48, 96, 192, 1024]
        elif net_type == 1:
            self.out_channels = [1, 24, 116, 232, 464, 1024]
        else:
            self.out_channels = [1, 24, 116, 232, 464, 1024]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.out_channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_channels[1]),
            nn.ReLU(inplace=True)
        )

        in_c = self.out_channels[1]
        self.stages = []
        for stage_idx in range(len(self.stage_repeat_num)):
            out_c = self.out_channels[2 + stage_idx]
            repeat_num = self.stage_repeat_num[stage_idx]
            for i in range(repeat_num):
                if i == 0:
                    self.stages.append(ShuffleBlock(in_c, out_c, downsample=True))
                else:
                    self.stages.append(ShuffleBlock(in_c, in_c, downsample=False))
                in_c = out_c
        self.stages = nn.Sequential(*self.stages)

        in_c = self.out_channels[-2]
        out_c = self.out_channels[-1]
        self.conv5 = conv_1x1_bn(in_c, out_c, 1)

        self.g_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.g_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ShuffleNet2Fer(num_classes=7, input_size=48, net_type=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    test_input = t.randn(1, 1, 48, 48)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
