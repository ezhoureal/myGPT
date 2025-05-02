import regex
from collections import Counter
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

def train_bpe(input_path, vocab_size, special_tokens: list[str]):
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

    for _ in range(vocab_size - 256 - len(special_tokens)):
        freqs = get_freq(tokens)
        if len(freqs) == 0:
            break
        tokens_to_merge: tuple[int, int] = max(freqs, key=freqs.get)
        byte_pair = (vocab[tokens_to_merge[0]], vocab[tokens_to_merge[1]])
        merges.append(byte_pair)
        new_id = len(vocab)
        vocab[new_id] = b"".join(byte_pair)
        new_tokens = []
        for word in tokens:
            new_chunk = merge(word, tokens_to_merge, new_id, freqs)
            new_tokens.append(new_chunk)
        tokens = new_tokens

    for special in special_tokens:
        vocab[len(vocab)] = special.encode()

    return (vocab, merges)

def encode(text: str, map: dict[int, bytes], merges: list[(int, int)]):
    b2t = {v: k for k, v in map.items()} # bytes to token map
    textBytes = text.encode()
    pos: dict[(int, int) : [int]] = {} # cache position of pairs
    for i in range(len(textBytes)) - 1:
        pair = (textBytes[i], textBytes[i + 1])
        pos.get(pair, []).append(i)
    for merge in merges:
        assert pos.get(merge)

if __name__ == "__main__":
    res = train_bpe("small_set.txt", 300, [])