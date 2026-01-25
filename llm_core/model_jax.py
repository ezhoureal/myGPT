from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
import optax


@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_head: int = 12
    n_layer: int = 12
    n_emb: int = 768  # hidden size
    dropout: float = 0
    use_kv_cache: bool = False  # this implies inference mode


class MLP(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.config.n_emb, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(self.config.n_emb, name='c_proj')(x)
        return x


class Attention(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # batch size, time (sequence length), channel (token embedding dimension)
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = nn.Dense(self.config.n_emb * 3, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Split into heads: (B, heads, seq, head_dim)
        head_dim = C // self.config.n_head
        q = q.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.config.n_head, head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        attention = (q @ jnp.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(head_dim))

        # Apply causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        attention = jnp.where(mask == 0, float('-inf'), attention)
        attention = jax.nn.softmax(attention, axis=-1)

        # Apply attention to values
        y = attention @ v  # (B, heads, seq, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        y = nn.Dense(self.config.n_emb, name='c_proj')(y)
        return y


class Layer(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, k_cache=None, v_cache=None, cache_len=0):
        # Pre-norm architecture
        ln_x = nn.LayerNorm(name='ln_1')(x)

        attn_out = Attention(self.config, name='attn')(ln_x)
        x = x + attn_out
        new_k_cache, new_v_cache = None, None
        # MLP block
        x = x + MLP(self.config, name='mlp')(nn.LayerNorm(name='ln_2')(x))

        return x, new_k_cache, new_v_cache


class GPT(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, start_idx: int = 0, k_caches=None, v_caches=None, cache_len=0):
        # x.shape = (B, T)
        B, T = x.shape
        assert T <= self.config.block_size

        # Token and position embeddings
        wte = nn.Embed(self.config.vocab_size, self.config.n_emb, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.n_emb, name='wpe')

        pos = jnp.arange(start_idx, start_idx + T)
        pos_emb = wpe(pos)  # (T, C)
        x_emb = wte(x) + pos_emb  # (B, T, C)

        # Transformer layers
        new_k_caches = []
        new_v_caches = []
        for i in range(self.config.n_layer):
            k_cache = k_caches[i] if k_caches is not None else None
            v_cache = v_caches[i] if v_caches is not None else None
            x_emb, new_k, new_v = Layer(self.config, name=f'h_{i}')(
                x_emb, k_cache, v_cache, cache_len)
            new_k_caches.append(new_k)
            new_v_caches.append(new_v)

        # Final layer norm
        z = nn.LayerNorm(name='ln_f')(x_emb)

        # Language model head (weight tied with token embeddings)
        # In Flax, we need to access the embedding weights explicitly
        logits = z @ wte.embedding.T

        return logits, new_k_caches, new_v_caches

    def compute_loss(self, params, x, target):
        """Compute cross-entropy loss."""
        logits, _, _ = self.apply({'params': params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1)
        ).mean()
        return loss

    @staticmethod
    def configure_optimizer(learning_rate: float, weight_decay: float):
        """Configure AdamW optimizer with weight decay."""
        # Create optimizer with weight decay
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay
        )
        return optimizer

    @staticmethod
    def init_weights(rng, config: Config, batch_size: int = 1, seq_len: int = 1):
        """Initialize model parameters."""
        model = GPT(config)
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy_input)
        return variables['params']


def generate(model, params, rng, prompt_tokens, max_new_tokens: int,
             temperature: float = 1.0, top_k: Optional[int] = None):
    """Generate text autoregressively."""
    tokens = jnp.array(prompt_tokens).reshape(1, -1)

    for _ in range(max_new_tokens):
        # Forward pass
        logits, _, _ = model.apply({'params': params}, tokens)
        logits = logits[:, -1, :] / temperature

        # Optional top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            logits = jnp.full_like(logits, float('-inf'))
            logits = logits.at[0, top_k_indices[0]].set(top_k_logits[0])

        # Sample next token
        rng, sample_rng = jax.random.split(rng)
        probs = jax.nn.softmax(logits, axis=-1)
        next_token = jax.random.categorical(sample_rng, jnp.log(probs), axis=-1)

        # Append to sequence
        tokens = jnp.concatenate([tokens, next_token.reshape(1, 1)], axis=1)

    return tokens[0].tolist()

def train():
    x = jax.random.randint(
        key=jax.random.PRNGKey(0), shape=(2, 10), minval=0, maxval=50304)
    y = jax.random.randint(
        key=jax.random.PRNGKey(1), shape=(2, 10), minval=0, maxval=50304)
    model = GPT(Config(vocab_size=50304))
    rng = jax.random.PRNGKey(0)
    params = GPT.init_weights(rng, Config(vocab_size=50304), batch_size=2, seq_len=10)
    
    loss_fn = jax.jit(lambda params: model.compute_loss(params, x, y))
    grad_fn = jax.grad(loss_fn)
    loss_fn(params)  # Initial loss computation
    grads = grad_fn(params)
    print("Training step completed. Loss and gradients computed.")
    params = jax.tree.map(lambda param, grad: param - 0.001 * grad, params, grads)
    print(f"params = {params}")

if __name__ == "__main__":
    train()