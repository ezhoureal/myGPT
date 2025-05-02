import pytest
from tokenizer import train_bpe

def test_train_bpe_empty_file(tmp_path):
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    vocab_size = 300
    special_tokens = ["<|endoftext|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(final_set) == 256 + len(special_tokens), "Final set should contain initial bytes and special tokens"
    assert len(merges) == 0, "No merges should occur for an empty file"
    assert len(set(final_set.values())) == len(final_set), "Final set values should be unique"

def test_train_bpe_basic_case(tmp_path):
    input_file = tmp_path / "basic.txt"
    input_file.write_text("hello world")
    vocab_size = 300
    special_tokens = ["<|endoftext|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(final_set) > 256, "Final set should contain more than initial bytes"
    assert len(merges) > 0, "Merges should occur for non-empty input"
    assert len(set(final_set.values())) == len(final_set), "Final set values should be unique"

def test_train_bpe_special_tokens(tmp_path):
    input_file = tmp_path / "special_tokens.txt"
    input_file.write_text("test data")
    vocab_size = 300
    special_tokens = ["<|endoftext|>", "<|pad|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    for token in special_tokens:
        assert token.encode() in final_set.values(), f"Special token {token} should be in final set"
    assert len(set(final_set.values())) == len(final_set), "Final set values should be unique"

def test_train_bpe_vocab_size_limit(tmp_path):
    input_file = tmp_path / "vocab_limit.txt"
    input_file.write_text("a b c d e f g h i j")
    vocab_size = 260  # Close to the initial byte size
    special_tokens = ["<|endoftext|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(final_set) == vocab_size, "Final set size should match the vocab size"
    assert len(merges) == vocab_size - 256 - len(special_tokens), "Number of merges should match the vocab size limit"
    assert len(set(final_set.values())) == len(final_set), "Final set values should be unique"

def test_train_bpe_no_special_tokens(tmp_path):
    input_file = tmp_path / "no_special_tokens.txt"
    input_file.write_text("simple test case")
    vocab_size = 300
    special_tokens = []

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(set(final_set.values())) == len(final_set), "Final set values should be unique"

from tokenizer import merge
def test_merge():
    tokens = [1, 2, 3, 4, 5]
    res = merge(tokens, (3, 4), 99, {})
    assert res == [1, 2, 99, 5]