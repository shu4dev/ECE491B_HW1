from train_bpe import run_train_bpe
from helper import save_voacb_and_merge

def bpe_tinystories(vocab_path: str, merges_path: str):
    vocab, merges = run_train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens=["<|endoftext|>"])
    save_voacb_and_merge(vocab, merges, vocab_path, merges_path)

if __name__ == "__main__":
    bpe_tinystories("ece496b_basics/Experiment_output/tinystories_vocab.json","ece496b_basics/Experiment_output/tinystories_merges.txt")