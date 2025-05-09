import torch

def naive_rope_cache(head_dim: int, seq_len: int, theta: float = 10000.0):
    assert head_dim % 2 == 0
    theta_numerator = torch.arange(0, head_dim, 2).float()
    thetas = 1.0 / theta ** (theta_numerator / head_dim)
    positions = torch.arange(seq_len).float().unsqueeze(1)
    angles = positions * thetas.unsqueeze(0)
    assert angles.shape == (seq_len, head_dim // 2)
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    ops = torch.stack((cosines, sines), dim=2)
    assert ops.shape == (seq_len, head_dim // 2, 2)
    return ops

class RotaryPositionalEmbedding(torch.nn.Module):
    @torch.no_grad
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        cache = naive_rope_cache(d_k, max_seq_len, theta)
        self.register_buffer("cache", cache, persistent=False)

    @torch.no_grad
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cache = self.get_buffer("cache")
        x = x.reshape(*x.shape[:-1], -1, 2)  # pair x along the head_dim

        # Adjust cache based on token_positions
        cache = cache[token_positions]  # (..., seq_len, d_k // 2, 2)
        output = torch.stack([
            x[..., 0] * cache[..., 0] - x[..., 1] * cache[..., 1],  # x_1 * cos - x_2 * sin
            x[..., 0] * cache[..., 1] + x[..., 1] * cache[..., 0]   # x_1 * sin + x_2 * cos
        ], dim=-1)
        output = output.flatten(-2)
        return output

emb = RotaryPositionalEmbedding(10000, 256, 24)
x = torch.rand(5, 2, 12, 256) # B = (5, 2)
token_positions = torch.arange(5, 17).expand(5, 2, 12)
y = emb(x, token_positions)
assert y.shape == x.shape, y.shape
