import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, channels:int, heads:int, dropout = 0.2) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).transpose(1, 2)
        x_ln = self.ln(x)
        att, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + att

        x = x + self.ffn(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

