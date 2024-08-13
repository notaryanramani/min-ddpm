import torch 
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention

class Convs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        out = F.gelu(self.net(x))
        return out
    

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_embd = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            Convs(in_channels, out_channels),
            Convs(out_channels, out_channels),
        )

        self.time_net = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x, t):
        x = self.net(x)
        t = self.time_net(t)[:, :, None, None]
        out = x + t
        return out
    

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_embd=256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.net = nn.Sequential(
            Convs(in_channels, out_channels),
            Convs(out_channels, out_channels)
        )

        self.time_net = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_channels),
            nn.SiLU()
        )

    def forward(self, x, t, residual):
        x = self.upsample(x)
        x = torch.cat([x, residual], dim = 1)
        x = self.net(x)
        t = self.time_net(t)[:, :, None, None]
        out = x + t
        return out


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, layers=3, n_labels=10, channels = 64, n_heads=4, n_embd=256, time_steps=1000):
        super().__init__()
        self.layers = 3
        outs = [i for i in range(1, layers)] + [layers-1]
        ins = [i for i in range(layers)]

        self.time_encoding = nn.Embedding(time_steps, n_embd)
        self.label_encoding = nn.Embedding(n_labels, n_embd)

        # Neural Net
        self.in_conv = Convs(c_in, channels)

        self.down_convs = nn.ModuleList(
            [DownConv(channels * (2 ** in_c), channels * (2 ** out_c)) for in_c, out_c in zip(ins, outs)]
        )
        self.down_attentions = nn.ModuleList(
            [SelfAttention(channels * (2**out_c), n_heads) for out_c in outs]
        )

        self.bottleneck = nn.Sequential(
            Convs(channels * (2 ** (layers - 1)), channels * (2 ** layers)),
            Convs(channels * (2 ** layers ), channels * (2 ** layers)),
            Convs(channels * (2 ** layers ), channels * (2 ** (layers - 1)))
        )


        self.up_convs = nn.ModuleList(
            [UpConv(channels * (2 ** in_c) * 2,  channels * (2 ** (layers - out_c - 1))) for in_c, out_c in zip(ins[::-1], outs)]
        )
        
        self.up_attentions = nn.ModuleList(
            [SelfAttention(channels * (2 ** (layers - out_c - 1)), n_heads) for out_c in outs]
        )

        self.out_conv = nn.Conv2d(channels, c_out, kernel_size=1)
    

    def forward(self, x, t, y=None):
        t = self.time_encoding(t)
        if y is not None:
            y = self.label_encoding(y) 
            t += y # add label encodings if provided
        x_skips = []

        x = self.in_conv(x)
        x_skips.append(x) # append first conv

        for i, (net, att) in enumerate(zip(self.down_convs, self.down_attentions)):
            x = net(x, t)
            x = att(x)
            if i != self.layers - 1: # ignore last down conv for append
                x_skips.append(x)

        x = self.bottleneck(x)
    
        for i, (net, att) in enumerate(zip(self.up_convs, self.up_attentions)):
            x = net(x, t, x_skips.pop())
            x = att(x)

        x = self.out_conv(x)

        return x


