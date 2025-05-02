import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge(tokens: list[int], pair_to_merge: tuple[int, int], new_id, freqs):
    i = 0
    res = []
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]:
            res.append(new_id)
            if i > 0:
                freqs[(tokens[i - 1], tokens[i])] -= 1
            if i < len(tokens) - 2:
                freqs[(tokens[i + 1], tokens[i + 2])] -= 1
            i += 2

        else:
            res.append(tokens[i])
            i += 1

    # update freqs based on new tokens
    for i in range(len(res) - 1):
        if res[i] == new_id or res[i + 1] == new_id:
            pair = (res[i], res[i + 1])
            freqs[pair] = freqs.get(pair, 0) + 1
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
    words: list[list[int]] = []
    # init
    for word in pre_tokens:
        words.append([])
        for i in range(len(word)):
            if i < len(word) - 1:
                pair = (word[i], word[i + 1])
                freqs[pair] = freqs.get(pair, 0) + 1
            words[-1].append(word[i])

    vocab: dict[int, bytes] = { i : i.to_bytes() for i in range(256)}
    merges: list[(bytes, bytes)] = []

    for _ in range(vocab_size - 256 - len(special_tokens)):
        if len(freqs) == 0:
            break
        tokens_to_merge: tuple[int, int] = max(freqs.items(), key=lambda x: (x[1], x[0]))[0]
        byte_pair = (vocab[tokens_to_merge[0]], vocab[tokens_to_merge[1]])
        merges.append(byte_pair)
        new_id = len(vocab)
        vocab[new_id] = []
        freqs.pop(tokens_to_merge)
        new_words = []
        for word in words:
            new_word = merge(word, tokens_to_merge, new_id, freqs)
            new_words.append(new_word)
        words = new_words

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
    print(f'empty = {ord(' ')}')