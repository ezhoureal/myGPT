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
    def forward(self, x):
        # add kv cache later for inference
        qkv: torch.Tensor = self.qkv_weight(x) # (batch, length, emb)
        q, k, v = qkv.split(self.n_emb, dim=2)
        attention = q @ k.transpose() / math.sqrt(self.n_emb)
        v = self.vw(x)
        return x

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
