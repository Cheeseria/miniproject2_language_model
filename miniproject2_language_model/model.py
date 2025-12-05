import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        assert config.embed_dim % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads

        self.c_attn = nn.Linear(config.embed_dim, 3*config.embed_dim, bias=False)

        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.bool())

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape # B=batch,T=time,C=embed_dim
        
        qkv = self.c_attn(x) # B, T, 3C
        q, k, v = qkv.split(C,dim=-1) #each B,T,C

        #heads split
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm_1 = nn.LayerNorm(config.embed_dim)
        self.CausalSelfAttn = CausalSelfAttention(config)
        self.LayerNorm_2 = nn.LayerNorm(config.embed_dim)

        self.MLP = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU()
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.CausalSelfAttn(self.LayerNorm_1(x))
        x = x + self.MLP(self.LayerNorm_2(x))
        return x