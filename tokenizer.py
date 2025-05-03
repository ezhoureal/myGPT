import regex
from collections import Counter

import tqdm
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_freq(tokens: list[list[int]]):
    pairs: list[(int, int)] = []
    for token in tokens:
        for i in range(len(token) - 1):
            pairs.append((token[i], token[i + 1]))
    return Counter(pairs)

def merge(tokens: list[int], pair_to_merge: tuple[int, int], new_id, freqs):
    i = 0
    res = []
    cnt = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]:
            res.append(new_id)
            # if i > 0:
            #     freqs[(tokens[i - 1], tokens[i])] -= 1
            # if i < len(tokens) - 2:
            #     freqs[(tokens[i + 1], tokens[i + 2])] -= 1
            i += 2
            cnt += 1

        else:
            res.append(tokens[i])
            i += 1

    # update freqs based on new tokens
    # for i in range(len(res) - 1):
    #     if res[i] == new_id or res[i + 1] == new_id:
    #         pair = (res[i], res[i + 1])
    #         freqs[pair] = freqs.get(pair, 0) + 1
    # assert(cnt == freqs.get(pair_to_merge, 0)), f'cnt = {cnt}'
    # freqs.pop(pair_to_merge)
    return res

def train_bpe(input_path, vocab_size, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer on a given text corpus.
    Args:
        input_path (str): The file path to the input text corpus.
        vocab_size (int): The desired vocabulary size, including special tokens.
        special_tokens (list[str]): A list of special tokens to include in the vocabulary.
    Returns:
        tuple[dict[int, bytes], list[tuple[int, int]]]: 
            - A dictionary mapping token IDs to their corresponding byte sequences.
            - A list of merge operations represented as tuples of token IDs.
    Raises:
        AssertionError: If `vocab_size` is not greater than 0.
    Notes:
        - The function reads the input corpus, tokenizes it into byte sequences, and iteratively merges the most 
          frequent byte pairs until the desired vocabulary size is reached.
        - Special tokens are added to the vocabulary after the merging process.
        - The function assumes that the input corpus is encoded in UTF-8 and uses a regex pattern (`PAT`) 
          to tokenize the text.
        - The `get_most_frequent_pair` and `merge` helper functions are expected to be defined elsewhere in the code.
    """
    assert vocab_size > 0
    pre_tokens = None
    with open(input_path, 'r') as f:
        corpus = f.read()
        # todo: read in chunks
        pre_tokens: list[str] = regex.findall(PAT, corpus)
        pre_tokens: list[bytes] = [p.encode() for p in pre_tokens]

    if pre_tokens is None:
        return
    
    freqs: dict[(int, int): int] = {}
    tokens: list[list[int]] = []
    # init
    for word in pre_tokens:
        tokens.append([])
        for i in range(len(word)):
            if i < len(word) - 1:
                pair = (word[i], word[i + 1])
                freqs[pair] = freqs.get(pair, 0) + 1
            tokens[-1].append(word[i])

    vocab: dict[int, bytes] = { i : i.to_bytes() for i in range(256)}
    merges: list[(bytes, bytes)] = []

    for _ in tqdm.trange(vocab_size - 256 - len(special_tokens)):
        freqs = get_freq(tokens)
        if len(freqs) == 0:
            break
        tokens_to_merge: tuple[int, int] = max(freqs, key=freqs.get)
        merges.append(tokens_to_merge)
        new_id = len(vocab)
        vocab[new_id] = b''.join([vocab[tokens_to_merge[0]], vocab[tokens_to_merge[1]]])
        print(f'merging {vocab[tokens_to_merge[0]].decode()} and {vocab[tokens_to_merge[1]].decode()}, in token id = {tokens_to_merge}')
        new_tokens = []
        for word in tokens:
            new_chunk = merge(word, tokens_to_merge, new_id, freqs)
            new_tokens.append(new_chunk)
        tokens = new_tokens

    for special in special_tokens:
        vocab[len(vocab)] = special.encode()

    return (vocab, merges)

def encode(text: str, merges: list[tuple[int, int]]) -> bytes:
    text_bytes = text.encode()
    new_id = 256
    for merge in merges:
        pos: dict[(int, int) : [int]] = {} # cache position of pairs
        for i in range(len(text_bytes) - 1):
            pair = (text_bytes[i], text_bytes[i + 1])
            pos.setdefault(pair, []).append(i)

        if pos.get(merge) is None:
            new_id += 1
            continue
        to_merge = pos.get(merge)
        new_text = []
        i = 0
        while i < len(text_bytes):
            if i in to_merge:
                new_text.append(new_id)
                i += 2
            else:
                new_text.append(text_bytes[i])
                i += 1
        new_id += 1
        text_bytes = new_text
    return text_bytes

def decode(tokens: bytes, vocab: dict[int, bytes]) -> str:
    res = []
    for token in tokens:
        res.append(vocab.get(token, b''))
    return b''.join(res).decode()

if __name__ == "__main__":
    vocab, merges = train_bpe("small_set.txt", 300, ["SPECIAL"])
    text = "Hello world"
    encoded = encode(text, merges)
    decoded = decode(encoded, vocab)
    assert decoded == text, decoded
