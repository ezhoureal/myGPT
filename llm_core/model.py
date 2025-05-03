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
    dropout: float = 0
    use_kv_cache: bool = False # this implies inference mode

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

        # this runs flash attention implemented by PyTorch
        # y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class AttentionWithKVCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_emb, config.n_emb * 3)
        self.n_emb = config.n_emb
        self.n_head = config.n_head
        self.c_proj = nn.Linear(config.n_emb, config.n_emb)
        self.len = 0
        self.config = config

    def share_weights_with_attention(self, attention: Attention):
        self.c_attn.weight = attention.c_attn.weight
        self.c_attn.bias = attention.c_attn.bias
        self.c_proj.weight = attention.c_proj.weight
        self.c_proj.bias = attention.c_proj.bias

    @torch.no_grad # inference only
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        assert C == self.n_emb
        # prefill
        qkv:torch.Tensor = self.c_attn(x)
        q, k, v = qkv.split(self.n_emb, dim=2)
        if self.len == 0:
            # initialize kv cache
            self.k_cache = k.clone()
            self.v_cache = v.clone()
        else:
            assert T == 1
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_cache.view(B, T + self.len, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_cache.view(B, T + self.len, self.n_head, C // self.n_head).transpose(1, 2)
        attn: torch.Tensor = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        if (self.len == 0):
            mask = torch.tril(torch.ones(T, T + self.len)).view(1, 1, T, T + self.len)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        # (B, head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        self.len += T
        return y

class Layer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_emb)
        self.attn = Attention(config)
        if (config.use_kv_cache):
            self.kv_attn = AttentionWithKVCache(config)
        self.ln_2 = nn.LayerNorm(config.n_emb)
        self.mlp = MLP(config)

    def forward(self, x):
        ln = self.ln_1(x)
        if hasattr(self, "kv_attn"):
            attn = self.kv_attn(ln)
            # cmp = self.attn(ln)
            # assert attn.shape == cmp.shape
            # assert attn.equal(cmp)
        else:
            attn = self.attn(ln)
        x = x + attn
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
            nn.init.normal(module.weight, mean=0.0, std=0.02)
        elif module.isinstance(nn.Linear):
            std = 0.02
            # scale down std because residual connections are added to final result
            # so std would accumulate
            if hasattr(module, "RESIDUAL_STD"):
                std *= (2 * self.config.n_layer) ** -0.5 # times 2 because Attn + MLP counts as 2
            nn.init.normal(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # predicts the next token
    def forward(self, x, target, startIdx: int = 0):
        # x.shape = (B, T)
        B, T = x.shape
        assert T <= self.config.block_size
        pos = torch.arange(start=startIdx, end=startIdx + T, device=x.device)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        x_emb = self.transformer.wte(x) + pos_emb # (B, T, C)
        for layer in self.transformer.h:
            x_emb = layer(x_emb)
        z = self.transformer.ln_f(x_emb)
        assert z.shape == (B, T, self.config.n_emb)
        next = self.lm_head(z) # (B, T, C)
        loss = None
        if target is not None:
            loss = nn.functional.cross_entropy(next.view(-1, next.shape[-1]), target.view(-1))
        return next, loss
    
    def configure_optimizer(self, weight_decay: int, learning_rate: int, device):
        params = [p for pn, p in self.named_parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2] # matrices
        non_decay_params = [p for p in params if p.dim() < 2]
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in torch.inspect.signature(torch.optim.AdamW).parameters
        fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if fused else dict()
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, **extra_args)
        print(f'fused available = {fused}')
        return optimizer
    @classmethod
    def from_pretrained(cls, config: Config):
        from transformers import AutoTokenizer, GPT2LMHeadModel

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys()if not k.endswith(".attn.mask")]
        sd_keys = [k for k in sd_keys if k.count(".kv_attn.") == 0]
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model_hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        hf_sd = model_hf.state_dict()
        hf_keys = hf_sd.keys()

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert sd_keys.__len__() == hf_keys.__len__(), sd_keys
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
        if config.use_kv_cache:
            layers: nn.ModuleList = model.transformer.h
            for layer in layers:
                assert isinstance(layer, Layer)
                layer.kv_attn.share_weights_with_attention(layer.attn)
        return model
