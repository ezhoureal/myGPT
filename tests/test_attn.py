import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import Config, Attention, AttentionWithKVCache

def test_attention_output_shape():
    config = Config(block_size=16, n_emb=32, n_head=4)
    attention = Attention(config)
    x = torch.randn(2, 16, 32)  # Batch size 2, sequence length 16, embedding size 32
    output = attention(x)
    assert output.shape == (2, 16, 32), f"Expected shape (2, 16, 32), got {output.shape}"

def test_attention_with_kv_cache_output_shape():
    config = Config(block_size=16, n_emb=32, n_head=4, use_kv_cache=True)
    attention = Attention(config)
    kv_attention = AttentionWithKVCache(config)
    kv_attention.share_weights_with_attention(attention)

    x = torch.randn(2, 16, 32)  # Batch size 2, sequence length 16, embedding size 32
    output = kv_attention(x)
    assert output.shape == (2, 16, 32), f"Expected shape (2, 16, 32), got {output.shape}"

def test_kv_cache_behavior():
    config = Config(block_size=16, n_emb=32, n_head=4, use_kv_cache=True)
    kv_attention = AttentionWithKVCache(config)

    # First forward pass (prefill)
    x1 = torch.randn(2, 8, 32)  # Batch size 2, sequence length 8, embedding size 32
    output1 = kv_attention(x1)
    assert kv_attention.len == 8, f"Expected cache length 8, got {kv_attention.len}"

    # Second forward pass (incremental)
    x2 = torch.randn(2, 1, 32)  # Batch size 2, sequence length 1
    output2 = kv_attention(x2)
    assert kv_attention.len == 9, f"Expected cache length 9, got {kv_attention.len}"

    # Check that the output shapes are correct
    assert output1.shape == (2, 8, 32), f"Expected shape (2, 8, 32), got {output1.shape}"
    assert output2.shape == (2, 1, 32), f"Expected shape (2, 1, 32), got {output2.shape}"

def test_kv_cache_matches_attention():
    config = Config(block_size=32, n_emb=32, n_head=4, use_kv_cache=True)
    attention = Attention(config)
    kv_attention = AttentionWithKVCache(config)
    kv_attention.share_weights_with_attention(attention)

    # prefill
    x = torch.randn(2, 16, 32)  # Batch size 2, sequence length 16, embedding size 32
    output_attention = attention(x)
    output_kv_attention = kv_attention(x)
    assert torch.allclose(output_attention, output_kv_attention, atol=1e-5), "Outputs of Attention and AttentionWithKVCache do not match"

    # decode
    for i in range(16):
        inc = torch.randn(2, 1, 32)
        x = torch.cat((x, inc), dim=1)
        output_attention = attention(x)[:, -1, :]
        output_kv_attention = kv_attention(inc).squeeze(1)
        assert output_attention.shape == output_kv_attention.shape
        assert torch.allclose(output_attention, output_kv_attention, atol=1e-5), "Outputs of Attention and AttentionWithKVCache do not match"
