from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import json

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=32000,
    special_tokens=[]
)
files = ["ECE491B_HW1/data/owt_train.txt"]

tokenizer.train(files, trainer=trainer)

vocab = tokenizer.get_vocab()  
with open("ECE491B_HW1/ece496b_basics/Experiment_output/HF.owt_vocab.json", "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=2)


tokenizer_json = tokenizer.to_str()
tokenizer_dict = json.loads(tokenizer_json)
merges = tokenizer_dict["model"]["merges"]


with open("ECE491B_HW1/ece496b_basics/Experiment_output/HF.owt_merges_txt", "w", encoding="utf-8") as merge_file:
    for merge in merges:
        if isinstance(merge, list):
            merge_line = " ".join(merge)
        else:
            merge_line = merge
        merge_file.write(merge_line + "\n")