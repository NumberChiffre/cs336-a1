from collections import defaultdict
from typing import NamedTuple


class BPETokenizerParams(NamedTuple):
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


def train_bpe(string: str, num_merges: int, special_tokens: list[str]) -> BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1
        # Find the most common pair.
        pair = max(counts, key=counts.get)
        index1, index2 = pair
        # Merge that pair.
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)
    return BPETokenizerParams(vocab=vocab, merges=merges)
