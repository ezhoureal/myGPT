import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge(tokens: list[int], pair_to_merge: tuple[int, int], new_id, freqs):
    i = 0
    res = []
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]:
            res.append(new_id)
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

freqs: dict[(int, int): int] = {}
res = merge([1, 2, 3, 4, 5], (3, 4), 99, {})
print(res)

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

    final_set: dict[int, bytes] = { i : i for i in range(256)}
    merges: list[(bytes, bytes)] = []

    for _ in range(vocab_size - 256 - len(special_tokens)):
        if len(freqs) == 0:
            break
        most_frequent = max(freqs.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append((final_set[most_frequent[0]], final_set[most_frequent[1]]))
        new_id = len(final_set)
        final_set[new_id] = most_frequent
        freqs.pop(most_frequent)
        new_words = []
        for word in words:
            new_word = merge(word, most_frequent, new_id, freqs)
            new_words.append(new_word)
        words = new_words

    for special in special_tokens:
        final_set[len(final_set)] = special.encode()

    return (final_set, merges)