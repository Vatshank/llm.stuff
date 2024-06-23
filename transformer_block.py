import torch
from torch import nn
import torch.nn.functional as F

import cpp_funcs


class SelfAttention(nn.Module):
    """
    Calculate self-attention given input X of shape (batch_size, seq_len, D).
    This is inference only, we pass in the weights during init.
    """
    def __init__(self, D=128, heads=8):
        self.D = D
        self.heads = heads
        assert D % heads == 0, "Total hidden dimension should be divisible by number of heads"
        self.D_head = D // heads

    def forward(self, x, w_Q, w_K, w_V, w_out):
        """
        x: (batch_size, seq_len, D)
        w_Q, w_K, w_V: (D, D)
        """
        # w_proj = torch.cat([w_Q, w_K, w_V], dim=1)
        # qkv = x @ w_proj
        B, N, D = x.size()
        q, k, v = x @ w_Q, x @ w_K, x @ w_V
        q = q.view(B, N, self.heads, D//self.heads).transpose(1, 2)
        k = k.view(B, N, self.heads, D//self.heads).transpose(1, 2)
        v = v.view(B, N, self.heads, D//self.heads).transpose(1, 2)

        scale = 1 / (D // self.heads) ** 0.5
        att = q @ k.transpose(-1, -2) * scale

        att = F.softmax(att, dim=-1)
        out = att @ v

        # consolidate all heads
        # TODO: why contiguous here and not at the end?
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = out @ w_out
        return out
