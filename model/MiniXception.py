import torch
import torch.nn as nn

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def SeparableConv2D(in_channels, out_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=1, groups=in_channels, padding=1, bias=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
    )

class ResidualXceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(ResidualXceptionBlock, self).__init__()
        self.depthwise_conv1 = SeparableConv2D(in_channels, out_channels, kernel)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.depthwise_conv2 = SeparableConv2D(out_channels, out_channels, kernel)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.residual_bn(self.residual_conv(x))

        x = self.relu1(self.bn1(self.depthwise_conv1(x)))
        x = self.bn2(self.depthwise_conv2(x))
        x = self.maxpool(x)
        
        return x + residual

class Mini_Xception(nn.Module):
    def __init__(self):
        super(Mini_Xception, self).__init__()
        self.conv1 = conv_bn_relu(1, 8, kernel_size=3, stride=1, padding=0)
        self.conv2 = conv_bn_relu(8, 8, kernel_size=3, stride=1, padding=0)
        
        self.residual_blocks = nn.ModuleList([
            ResidualXceptionBlock(8 , 16),
            ResidualXceptionBlock(16, 32),
            ResidualXceptionBlock(32, 64),
            ResidualXceptionBlock(64, 128)            
        ])
        
        self.conv3 = nn.Conv2d(128, 7, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.global_avg_pool(self.conv3(x))
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    model = Mini_Xception()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.4f}M")
    
    test_input = torch.randn(1, 1, 48, 48)
    output = model(test_input)
    print(f"Output shape: {output.shape}")