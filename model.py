from dataclasses import dataclass
import math
import torch
import torch.nn as nn

@dataclass
class Config:
    block_size: int = 100
    vocab_size: int = 100
    n_head: int = 6
    n_layer: int = 5
    n_emb: int = 256 # hidden size

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.n_emb, 2 * config.n_emb)
        self.gelu = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(config.n_emb * 2, config.n_emb)
    def forward(self, x):
        x = self.linear(x)
        return self.proj(self.gelu(x))

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.qkv_weight = nn.Linear(config.n_emb, config.n_emb * 3)
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.proj = nn.Linear(config.n_emb, config.n_emb)
        T = config.block_size
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, time (sequence length), channel (token embedding dimension)
        # add kv cache later for inference
        qkv: torch.Tensor = self.qkv_weight(x) # (batch, length, emb)
        q, k, v = qkv.split(self.n_emb, dim=2)
        # split into heads: (B, heads, seq, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # (seq, headDim) @ (headDim, seq) = (seq, seq)
        attention = q @ k.transpose(-1, -2) / math.sqrt(self.n_emb)
        # only attend to previous tokens. do :T because T <= block_size
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        y = attention @ v # (seq, seq) @ (seq, headDim) = (seq, headDim)
        y = self.proj(y)
        return y

class Layer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_emb)
        self.ln2 = nn.LayerNorm()
        self.mlp = MLP(config)
        self.attn = Attention(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):
