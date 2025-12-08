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
            nn.GELU(), 
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.CausalSelfAttn(self.LayerNorm_1(x))
        x = x + self.MLP(self.LayerNorm_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.WTE = nn.EMbedding(config.vocab_size, config.embed_dim)
        self.WPE = nn.Parameter(torch.zeros(1, config.block_size, config.embed_dim))
        self.Drop = nn.Dropout(config.dropout)

        self.Blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])

        self.Final_LayerNorm = nn.LayerNorm(config.embed_dim)
        self.LM_Head = nn.Linear(config.embed_dim, config.vocab_size, bias = False)

        self.LM_Head.weight = self.WTE.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size

        tok_emb = self.WTE(idx)
        pos_emb = self.WPE[:, :T, :]
        x = self.Drop(tok_emb + pos_emb)

        for Block in self.Blocks:
            x = Block(x)

        x = self.Final_LayerNorm(x)
        logits = self.LM_Head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx