from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import tiktoken, tqdm

from data_loader import DataLoader

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_layer: int = 12
    n_emb: int = 768 # hidden size
    dropout: float = 0

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_emb, 4 * config.n_emb)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_emb * 4, config.n_emb)
        self.c_proj.RESIDUAL_STD = 1 # special marker
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)

class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_emb, config.n_emb * 3)
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        T = config.block_size
        self.register_buffer("mask", torch.tril(torch.ones(T, T, dtype=torch.float32)).view(1, 1, T, T))
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
        attention = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(k.size(-1)))
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
        self.transformer.wte.weight = self.lm_head.weight

    def init_weight(self, module: nn.Module):
        if module.isinstance(nn.Embedding):
            torch.nn.init.normal(module.weight, mean=0.0, std=0.02)
        elif module.isinstance(nn.Linear):
            std = 0.02
            # scale down std because residual connections are added to final result
            # so std would accumulate
            if hasattr(module, "RESIDUAL_STD"):
                std *= (2 * self.config.n_layer) ** -0.5 # times 2 because Attn + MLP counts as 2
            torch.nn.init.normal(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    # predicts the next token
    # x = all previous tokens
    def forward(self, x, target):
        # x.shape = (B, T)
        B, T = x.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, device=x.device)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        x_emb = self.transformer.wte(x) + pos_emb # (B, T, C)
        for layer in self.transformer.h:
            x_emb = layer(x_emb)
        z = self.transformer.ln_f(x_emb)
        assert z.shape == (B, T, self.config.n_emb)
        next = self.lm_head(z) # (B, T, C)
        loss = None
        if target is not None:
            loss = torch.nn.functional.cross_entropy(next.view(-1, next.shape[-1]), target.view(-1))
        return next, loss
    
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

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert sd_keys.__len__() == hf_keys.__len__()
        for k in sd_keys:
            assert hf_keys.__contains__(k)
            if any(k.endswith(postFix) for postFix in transposed):
                assert sd[k].shape == hf_sd[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert sd[k].shape == hf_sd[k].shape, f'key = {k}, sd shape = {sd[k].shape}, hf shape = {hf_sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
        return model

# Check if CUDA is available and set the device
device = (
    # torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f'using device {device}')

def inference(model: GPT):
    BATCH = 5
    MAX_LEN = 30
    K = 50
    PROMPT = "Hello, LLM. What's your name?"

    enc = tiktoken.encoding_for_model('gpt2')
    tokens = enc.encode(PROMPT)
    x = torch.tensor(tokens, dtype=torch.long, device=device)  # Move tensor to device
    x = x.unsqueeze(0).repeat(BATCH, 1)  # (B, T)

    for i in tqdm.trange(MAX_LEN):
        logits, _ = model(x, None)
        logits = logits[:, -1, :].squeeze(1)
        assert logits.shape == (BATCH, Config.vocab_size)
        probs = torch.softmax(logits, dim=-1)
        (top_probs, top_idx) = torch.topk(probs, K)
        next_idx = torch.multinomial(top_probs, 1)  # sample 1 from top_k
        assert next_idx.shape == (BATCH, 1)
        next_token = torch.gather(top_idx, dim=1, index=next_idx)
        assert next_token.shape == (BATCH, 1)
        x = torch.cat((x, next_token), dim=1)[:, -Config.block_size:]

    response = enc.decode_batch(x.tolist())
    print(f'output = {response}')

import time
def train():
    B = 6
    T = 1024
    model = GPT(Config()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    data = DataLoader(device)
    for i in tqdm.trange(10):
        x, y = data.get_batch(B, T)

        t1 = time.time()
        optimizer.zero_grad()
        logits, loss = model(x, y)
        print(f'loss = {loss}')
        loss.backward()
        optimizer.step()
        t2 = time.time()
        print(f'token processed speed = {B * T / (t2 - t1)}')

train()

# model = GPT.from_pretrained().to(device)
# inference()