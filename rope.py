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

cache = naive_rope_cache(4, 3)
print(cache)
x = torch.randn(3, 1, 4) # (seq_len, heads, head_dim)
x = x.reshape(*x.shape[:-1], -1, 2) # group x along the head_dim

print(x)

cache = cache.unsqueeze(1)
assert cache.shape == x.shape
output = torch.stack([
    x[..., 0] * cache[..., 0] - x[..., 1] * cache[..., 1],
    x[..., 0] * cache[..., 1] + x[..., 1] * cache[..., 0]
], dim=-1)
output = output.flatten(2)
assert output.shape == (3, 1, 4)
print(output)