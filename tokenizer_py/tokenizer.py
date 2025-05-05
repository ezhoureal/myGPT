import regex
from collections import Counter
import tqdm

# regular expression pattern taken from 
# The pattern is designed to match the following:
# - Contractions like 's, 'd, 'm, 't, 'll, 've, 're (e.g., "it's", "you've").
# - Words containing letters (\p{L}+), optionally preceded by a space.
# - Numbers (\p{N}+), optionally preceded by a space.
# - Non-whitespace, non-letter, non-number characters ([^\s\p{L}\p{N}]+), optionally preceded by a space.
# - Whitespace sequences (\s+) that are not followed by non-whitespace characters.
# The pattern ensures that text is split into meaningful tokens while preserving spaces and special characters.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_freq(tokens: list[list[int]]):
    pairs: list[(int, int)] = []
    for token in tokens:
        for i in range(len(token) - 1):
            pairs.append((token[i], token[i + 1]))
    return Counter(pairs)

def merge(tokens: list[int], pair_to_merge: tuple[int, int], new_id):
    i = 0
    res = []
    cnt = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]:
            res.append(new_id)
            i += 2
            cnt += 1

        else:
            res.append(tokens[i])
            i += 1
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
    with open(input_path, 'r') as f:
        corpus = f.read()
        # Split by special tokens
        segments = regex.split("|".join(special_tokens), corpus)
        # Further split each segment using PAT
        pre_tokens = []
        for segment in segments:
            matches = regex.finditer(PAT, segment)
            pre_tokens.extend([m.group().encode() for m in matches])

    if pre_tokens is None:
        return
    
    tokens: list[list[int]] = pre_tokens
    vocab: dict[int, bytes] = { i : i.to_bytes() for i in range(256)}
    merges: list[tuple[int, int]] = []

    freqs: dict[(int, int): int] = get_freq(tokens)
    for _ in tqdm.trange(vocab_size - 256 - len(special_tokens)):
        if len(freqs) == 0:
            break
        tokens_to_merge: tuple[int, int] = max(freqs, key=freqs.get)
        merges.append(tokens_to_merge)
        new_id = len(vocab)
        vocab[new_id] = b''.join([vocab[tokens_to_merge[0]], vocab[tokens_to_merge[1]]])
        new_tokens = []
        for chunk in tokens:
            new_chunk = merge(chunk, tokens_to_merge, new_id)
            new_tokens.append(new_chunk)
        tokens = new_tokens
        freqs.pop(tokens_to_merge)
        update_freq(freqs, tokens, new_id)

    for special in special_tokens:
        vocab[len(vocab)] = special.encode()

    return (vocab, merges)

def update_freq(freqs: dict[(int, int): int], tokens, new_id):
    for chunk in tokens:
        for i, token in enumerate(chunk):
            if token != new_id:
                continue
            if i > 0:
                prev_pair = (chunk[i - 1], new_id)
                freqs[prev_pair] = freqs.get(prev_pair, 0) + 1
            if i < len(chunk) - 1:
                next_pair = (new_id, chunk[i + 1])
                freqs[next_pair] = freqs.get(next_pair, 0) + 1

def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids = list(map(int, text.encode()))
    new_id = 256
    for merge_info in merges:
        ids = merge(ids, merge_info, new_id)
        new_id += 1
    return ids

def decode(tokens: list[int], vocab: dict[int, bytes]) -> str:
    res = []
    for token in tokens:
        res.append(vocab.get(token, b''))
    return b''.join(res).decode()

if __name__ == "__main__":
    vocab, merges = train_bpe("data/small_set.txt", 300, ["SPECIAL"])
    text = "Hello world"
    encoded = encode(text, merges)
    decoded = decode(encoded, vocab)
    assert decoded == text, decoded

    import rs_tokenizer as rs_tok
    tok = rs_tok.PyTokenizer("data/small_set.txt", ["SPECIAL"], 300)
    rs_encoded = tok.decode(tok.encode(text))
    assert rs_encoded == decoded
