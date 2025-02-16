import os
import time
import tracemalloc
from typing import List
import regex as re
from contextlib import contextmanager

# Context manager for per-section profiling (time & memory)
@contextmanager
def profile_section(section_name: str):
    start_time = time.perf_counter()
    snapshot_before = tracemalloc.take_snapshot()
    yield
    snapshot_after = tracemalloc.take_snapshot()
    end_time = time.perf_counter()
    time_taken = end_time - start_time
    # Calculate net memory difference (in bytes) over all traced lines.
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    mem_diff = sum(stat.size_diff for stat in stats)
    print(f"Section '{section_name}': Time taken: {time_taken:.4f} s, Memory diff: {mem_diff / 1024:.2f} KB\n")

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str] = None
):
    if special_tokens is None:
        special_tokens = []

    # Start overall memory tracking.
    tracemalloc.start()
    overall_start_time = time.perf_counter()

    # --- Section 1: Tokenization & Frequency Building ---
    with profile_section("Tokenization and Frequency Building"):
        word_freq = {}
        words_list = [] 
        freqs = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        with open(input_path, "r") as f:
            for line in f:
                tokens = re.findall(PAT, line)
                for token in tokens:
                    word_freq[tuple(token)] = word_freq.get(tuple(token), 0) + 1
        for (word, freq) in word_freq.items():
            words_list.append(list(word))
            freqs.append(freq)

    # --- Section 2: Building Pair Statistics ---
    with profile_section("Building Pair Statistics"):
        def build_pair_stats(words_list: List[List[str]], freqs: List[int]):
            pair2freq = {}
            for w_id, word in enumerate(words_list):
                frequency = freqs[w_id]
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    if pair not in pair2freq:
                        pair2freq[pair] = 0
                    pair2freq[pair] += frequency
            return pair2freq

        current_vocab_size = 256 + len(special_tokens)
        num_merges_to_do = max(0, vocab_size - current_vocab_size)
        pair2freq = build_pair_stats(words_list, freqs)
        merges_performed = []

    # --- Section 3: Merge Loop ---
    with profile_section("Merge Loop"):
        for _ in range(num_merges_to_do):
            if not pair2freq:
                break
            best_pair_count = 0
            best_pair = None
            for pair, counts in pair2freq.items():
                if counts > best_pair_count:
                    best_pair = pair
                    best_pair_count = counts
                if best_pair_count == 0 and best_pair is None:
                    break
                if counts == best_pair_count:
                    best_pair = max(best_pair, pair)
            if best_pair_count < 1:
                break    

            merges_locations = []
            for w_id, word in enumerate(words_list):
                for idx in range(len(word) - 1):
                    if (word[idx], word[idx + 1]) == best_pair:
                        merges_locations.append((w_id, idx))
                        if idx >= 1:
                            left_pair = (word[idx - 1], word[idx])
                            pair2freq[left_pair] = pair2freq.get(left_pair, 0) - freqs[w_id]
                        if idx + 2 < len(word):
                            right_pair = (word[idx + 1], word[idx + 2])
                            pair2freq[right_pair] = pair2freq.get(right_pair, 0) - freqs[w_id]

            sorted_locations = sorted(merges_locations, key=lambda x: (x[0], x[1]), reverse=True)

            for w_id, idx in sorted_locations:
                words_list[w_id][idx] = ''.join(best_pair)
                del words_list[w_id][idx + 1]
                new_word = words_list[w_id]
                if idx >= 1:
                    new_left_pair = (new_word[idx - 1], new_word[idx])
                    pair2freq[new_left_pair] = pair2freq.get(new_left_pair, 0) + freqs[w_id]
                if idx + 1 < len(new_word):
                    new_right_pair = (new_word[idx], new_word[idx + 1])
                    pair2freq[new_right_pair] = pair2freq.get(new_right_pair, 0) + freqs[w_id]

            pair2freq.pop(best_pair)
            merges_performed.append((best_pair[0].encode("utf-8"), best_pair[1].encode("utf-8")))
            current_vocab_size += 1
            if current_vocab_size >= vocab_size:
                break

    # --- Section 4: Final Vocabulary Construction ---
    with profile_section("Final Vocabulary Construction"):
        vocab = {}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        for i in range(256):
            vocab[len(vocab)] = bytes([i])
        for token in merges_performed:
            vocab[len(vocab)] = b''.join(token)

    overall_end_time = time.perf_counter()
    overall_time = overall_end_time - overall_start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Overall run time: {overall_time:.4f} seconds")
    print(f"Overall peak memory usage: {peak / 1024:.2f} KB\n")
    
    return vocab, merges_performed