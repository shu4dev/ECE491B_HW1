from train_bpe import run_train_bpe
from helper import save_voacb_and_merge

def bpe_tinystories(vocab_path: str, merges_path: str):
    vocab, merges = run_train_bpe("ECE491B_HW1/data/owt_train.txt", 32000, special_tokens=["<|endoftext|>"])
    save_voacb_and_merge(vocab, merges, vocab_path, merges_path)

if __name__ == "__main__":
    bpe_tinystories("ECE491B_HW1/ece496b_basics/Experiment_output/owt_vocab.json","ECE491B_HW1/ece496b_basics/Experiment_output/owt_merges.txt")