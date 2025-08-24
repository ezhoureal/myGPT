from tokenizer_py.tokenizer import encode, decode
from tokenizer_py.tokenizer import merge
import pytest
from tokenizer_py.tokenizer import train_bpe


def test_train_bpe_empty_file(tmp_path):
    input_file = tmp_path / "empty.txt"
    input_file.write_text("")
    vocab_size = 300
    special_tokens = ["<|endoftext|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(final_set) == 256 + \
        len(special_tokens), "Final set should contain initial bytes and special tokens"
    assert len(merges) == 0, "No merges should occur for an empty file"
    assert len(set(final_set.values())) == len(
        final_set), "Final set values should be unique"


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
    assert len(set(final_set.values())) == len(
        final_set), "Final set values should be unique"


def test_train_bpe_special_tokens(tmp_path):
    input_file = tmp_path / "special_tokens.txt"
    input_file.write_text("test<|endoftext|> data")
    vocab_size = 300
    special_tokens = ["<|endoftext|>", "<|pad|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    for token in special_tokens:
        assert token.encode() in final_set.values(
        ), f"Special token {token} should be in final set"
    assert len(set(final_set.values())) == len(
        final_set), "Final set values should be unique"


def test_train_bpe_vocab_size_limit(tmp_path):
    input_file = tmp_path / "vocab_limit.txt"
    input_file.write_text("a b c d e f g h i j")
    vocab_size = 260  # Close to the initial byte size
    special_tokens = ["<|endoftext|>"]

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(
        final_set) == vocab_size, "Final set size should match the vocab size"
    assert len(merges) == vocab_size - 256 - \
        len(special_tokens), "Number of merges should match the vocab size limit"
    assert len(set(final_set.values())) == len(
        final_set), "Final set values should be unique"


def test_train_bpe_no_special_tokens(tmp_path):
    input_file = tmp_path / "no_special_tokens.txt"
    input_file.write_text("simple test case")
    vocab_size = 300
    special_tokens = []

    result = train_bpe(str(input_file), vocab_size, special_tokens)
    assert result is not None, "Result should not be None"
    final_set, merges = result
    assert len(set(final_set.values())) == len(
        final_set), "Final set values should be unique"


def test_merge():
    tokens = [1, 2, 3, 4, 5]
    res = merge(tokens, (3, 4), 99)
    assert res == [1, 2, 99, 5]


def test_encode_no_merges():
    merges = []  # No merges applied
    text = "test"
    encoded = encode(text, merges)
    assert encoded == list(map(int, text.encode())
                           ), f"Encoded output mismatch: {encoded}"


def test_encode_special_characters():
    merges = [(33, 33)]  # Example merge for "!!"
    text = "hello!!"
    encoded = encode(text, merges)
    assert encoded[-1] == 256, f"Encoded output mismatch: {encoded}"


def test_decode_basic_case():
    vocab = {256: b"he", 257: b"llo"}
    tokens = [256, 257]
    decoded = decode(tokens, vocab)
    assert decoded == "hello", f"Decoded output mismatch: {decoded}"


def test_decode_no_vocab_match():
    vocab = {256: b"he", 257: b"llo"}
    tokens = [258]  # Token not in vocab
    decoded = decode(tokens, vocab)
    assert decoded == "", f"Decoded output mismatch: {decoded}"


def test_decode_with_special_tokens():
    vocab = {256: b"he", 257: b"llo", 258: b"<|endoftext|>"}
    tokens = [256, 257, 258]
    decoded = decode(tokens, vocab)
    assert decoded == "hello<|endoftext|>", f"Decoded output mismatch: {decoded}"


def test_integrated():
    vocab, merges = train_bpe("data/small_set.txt", 300, ["SPECIAL"])
    text = "Hello world"
    encoded = encode(text, merges)
    decoded = decode(encoded, vocab)
    assert decoded == text, decoded
