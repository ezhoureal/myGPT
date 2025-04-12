from dataclasses import dataclass
import math
import torch
import torch.nn as nn

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_layer: int = 12
    n_emb: int = 768 # hidden size

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_emb * 4, config.n_emb)
    def forward(self, x):
        x = self.c_fc(x)
        return self.c_proj(self.gelu(x))

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_emb, config.n_emb * 3)
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        T = config.block_size
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # batch size, time (sequence length), channel (token embedding dimension)
        # add kv cache later for inference
        qkv: torch.Tensor = self.c_attn(x) # (batch, length, emb)
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
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Layer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_emb),
            wpe = nn.Embedding(config.block_size, config.n_emb),
            h = nn.ModuleList([Layer(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_emb)
        ))
        self.lm_head = nn.Linear(config.n_emb, config.vocab_size, bias=False)

    # predicts the next token
    # x = all previous tokens
    def forward(self, x):
        # x.shape = (B, T)
        B, T = x.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, x.shape[1])
        pos_emb = self.transformer.wpe(pos) # (T, C)
        x_emb = self.transformer.wte(x) + pos_emb # (B, T, C)
        for layer in self.transformer.h:
            x_emb = layer(x_emb)
        z = self.transformer.ln_f(x_emb)
        assert z.shape == (B, T, self.config.n_emb)
        next = self.lm_head(z)
        return next
    
    @classmethod
    def from_pretrained(cls):
        from transformers import AutoTokenizer, GPT2LMHeadModel

        model = GPT(Config())
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys()if not k.endswith(".attn.mask")]
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model_hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        hf_sd = model_hf.state_dict()
        hf_keys = hf_sd.keys()

        keys_to_transpose = {".attn.c_attn.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"}
        assert sd_keys.__len__() == hf_keys.__len__()
        for k in sd_keys:
            if any(k.endswith(postFix) for postFix in keys_to_transpose):
                assert sd[k].shape == hf_sd[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert sd[k].shape == hf_sd[k].shape, f'key = {k}, sd shape = {sd[k].shape}, hf shape = {hf_sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
        return model

# using a fixed config for now
model = GPT.from_pretrained()
model.forward(torch.arange(0, 5).view(5, 1))