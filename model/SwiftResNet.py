import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CoordAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        mid_channels = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)                    
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class GhostConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1,
                 ratio: int = 2, dw_size: int = 5):
        super().__init__()
        self.out_c = out_c
        self.ratio = ratio
        primary_c = out_c // ratio
        ghost_c = out_c - primary_c

        self.primary = nn.Sequential(
            nn.Conv2d(in_c, primary_c, kernel_size, stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(primary_c),
            nn.Hardswish()
        )

        groups_cheap = primary_c if ghost_c % primary_c == 0 else 1
        self.cheap = nn.Sequential(
            nn.Conv2d(primary_c, ghost_c, dw_size, 1,
                      padding=dw_size // 2, groups=groups_cheap, bias=False),
            nn.BatchNorm2d(ghost_c),
            nn.Hardswish()
        )

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)


class LiteResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1,
                 expand_ratio: int = 3, dilation: int = 1, drop_path_rate: float = 0.0,
                 use_ghost: bool = True, use_attention: bool = True):
        super().__init__()
        self.stride = stride
        self.use_res_connect = (stride == 1 and in_c == out_c)
        self.drop_path = DropPath(drop_path_rate)

        hidden_dim = int(round(in_c * expand_ratio)) if expand_ratio != 1 else in_c

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish()
            ])

        if use_ghost:
            layers.append(
                GhostConv(hidden_dim, hidden_dim, kernel_size=kernel_size,
                          stride=stride, ratio=2, dw_size=5)
            )
        else:
            padding = (kernel_size - 1) * dilation // 2 if stride == 1 else kernel_size // 2
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                          padding=padding, groups=hidden_dim, dilation=dilation, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish()
            ])

        if use_attention:
            layers.append(CoordAttention(hidden_dim, reduction=16))

        layers.extend([
            nn.Conv2d(hidden_dim, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


class SwiftResNet(nn.Module):
    def __init__(self, config: dict, num_classes: int = 7):
        super().__init__()
        self.use_simple_fusion = config.get('use_simple_fusion', True)

        stem_width = config['stem_width']

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_width // 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width // 2),
            nn.Hardswish(),
            nn.Conv2d(stem_width // 2, stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.Hardswish(),
        )

        self.blocks = nn.ModuleList()
        input_channel = stem_width

        total_blocks = sum(num_blocks for _, num_blocks, _, _ in config['stages'])
        self.drop_path_rate = config.get('drop_path_rate', 0.0)
        block_idx = 0

        for stage_idx, (out_c, num_blocks, stride, expand_ratio) in enumerate(config['stages']):
            stage_layers = []
            dilation = 2 if stage_idx >= 2 else 1  

            drop_prob = self.drop_path_rate * block_idx / max(total_blocks - 1, 1)
            stage_layers.append(LiteResidualBlock(
                input_channel, out_c, stride=stride, expand_ratio=expand_ratio,
                dilation=dilation, drop_path_rate=drop_prob,
                use_ghost=True, use_attention=True
            ))
            input_channel = out_c
            block_idx += 1

            for _ in range(1, num_blocks):
                drop_prob = self.drop_path_rate * block_idx / max(total_blocks - 1, 1)
                stage_layers.append(LiteResidualBlock(
                    input_channel, out_c, stride=1, expand_ratio=expand_ratio,
                    dilation=dilation, drop_path_rate=drop_prob,
                    use_ghost=True, use_attention=True
                ))
                block_idx += 1

            self.blocks.append(nn.Sequential(*stage_layers))

        last_channel = input_channel
        head_dim = config.get('head_dim', last_channel)

        self.final_conv = nn.Sequential(
            nn.Conv2d(last_channel, head_dim, 1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.Hardswish()
        )

        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(head_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_single(self, x):
        x = self.stem(x)
        for stage in self.blocks:
            x = stage(x)
        x = self.final_conv(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        x = self.dropout(x)
        feat = x
        x = self.classifier(x)
       
        return feat, x

    def forward(self, x):
        if x.dim() == 5: 
            B, n_crops, C, H, W = x.shape
            x = x.view(B * n_crops, C, H, W)
            feats, out = self.forward_single(x)

            out = out.view(B, n_crops, -1)
            final_out = out.mean(dim=1)

            avg_feats = feats.view(B, n_crops, -1).mean(dim=1)
            return avg_feats, final_out
        
        return self.forward_single(x)


def get_extractnet_config(scale: str = 'tiny'):
    configs = {
        'micro': {
            'stem_width': 12,
            'stages': [
                [16, 2, 2, 2],   
                [24, 3, 2, 2],
                [48, 4, 2, 3],
                [80, 2, 1, 3],
            ],
            'head_dim': 192,
            'dropout': 0.15,
            'drop_path_rate': 0.05
        },
        'nano': {
            'stem_width': 16,
            'stages': [
                [24, 2, 2, 2],
                [32, 3, 2, 3],
                [64, 5, 2, 3],
                [96, 3, 1, 4],
            ],
            'head_dim': 256,
            'dropout': 0.15,
            'drop_path_rate': 0.08
        },
        'tiny': {
            'stem_width': 20,
            'stages': [
                [32, 3, 2, 3],
                [48, 4, 2, 3],
                [96, 6, 2, 4],
                [128, 3, 1, 4],
            ],
            'head_dim': 512,
            'dropout': 0.20,
            'drop_path_rate': 0.12
        },
        'small': {
            'stem_width': 24,
            'stages': [
                [40, 3, 2, 3],
                [64, 4, 2, 3],
                [128, 7, 2, 4],
                [160, 4, 1, 4],
            ],
            'head_dim': 640,
            'dropout': 0.22,
            'drop_path_rate': 0.15
        },
    }
    if scale not in configs:
        raise ValueError(f"Unknown scale: {scale}. Available: {list(configs.keys())}")
    return configs[scale]


def extract_net(scale: str = 'tiny', num_classes: int = 7):
    config = get_extractnet_config(scale)
    model = SwiftResNet(config, num_classes=num_classes)
    return model


if __name__ == "__main__":
    scales = ['micro', 'nano', 'tiny', 'small', 'base', 'large', 'xlarge']
    dummy_input = torch.randn(4, 5, 1, 48, 48)   

    print(f"{'Scale':<10} | {'Params':<15} | {'Model Size (MB)':<15} | {'Output Shape'}")
    print("-" * 72)
    for s in scales:
        model = extract_net(s)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = params * 4 / (1024 * 1024)
        out = model(dummy_input)
        print(f"{s.capitalize():<10} | {params/1e6:6.2f} M | {model_size_mb:6.2f} MB | {out.shape}")
