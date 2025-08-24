import os
import time
import pytest
from tokenizer_py.tokenizer import train_bpe, encode


@pytest.fixture
def test_file():
    return "data/small_set.txt"


def test_tokenizer_speed(test_file):
    TOTAL_BYTES = os.path.getsize(test_file)
    encoding_time = 0
    start_time = time.time()
    vocab, merges = train_bpe(test_file, 512, ["<END_OF_TEXT>"])
    end_time = time.time()
    training_time = end_time - start_time
    print(
        f'python tokenizer training speed = {
            TOTAL_BYTES /
            training_time /
            256} per step')

    encoding_time = 0
    with open(test_file) as f:
        for line in f:
            start_time = time.time()
            encode(line, merges)  # Replace with the actual encoding method
            end_time = time.time()
            encoding_time += end_time - start_time
    print(f'python tokenizer encoding speed = {TOTAL_BYTES / encoding_time}')


def test_rs_tokenizer_speed(test_file):
    import rs_tokenizer as rs_tok
    TOTAL_BYTES = os.path.getsize(test_file)
    start_time = time.time()
    tok = rs_tok.PyTokenizer(test_file, ["<END_OF_TEXT>"], 512)
    end_time = time.time()
    training_time = end_time - start_time
    print(
        f'rust tokenizer training speed = {
            TOTAL_BYTES /
            training_time /
            256} per step')

    encoding_time = 0
    with open(test_file) as f:
        for line in f:
            start_time = time.time()
            # Replace with the actual encoding method
            encoded = tok.encode(line)
            end_time = time.time()
            encoding_time += end_time - start_time
            assert line == tok.decode(encoded)
    print(f'rust tokenizer encoding speed = {TOTAL_BYTES / encoding_time}')


if __name__ == "__main__":
    pytest.main()
