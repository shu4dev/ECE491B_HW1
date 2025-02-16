import numpy as np
import tiktoken

def encode_and_serialize_dataset(encoding, input_file, output_file):
    token_ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = encoding.encode(line, allowed_special={"<|endoftext|>"})
                token_ids.extend(tokens)
    token_ids_array = np.array(token_ids, dtype=np.uint16)
    np.save(output_file, token_ids_array)
    print(f"Saved {len(token_ids)} tokens from '{input_file}' to '{output_file}'.")


tiny_stories_encoding = tiktoken.get_encoding("gpt2")
open_webtext_encoding = tiktoken.get_encoding("gpt2")

tinystories_train_file = 'data/TinyStoriesV2-GPT4-train.txt'
tinystories_dev_file   = 'data/TinyStoriesV2-GPT4-valid.txt'
tinystories_train_out  = 'ece496b_basics/Experiment_output/tinystories_train_tokens.npy'
tinystories_dev_out    = 'ece496b_basics/Experiment_output/tinystories_valid_tokens.npy'

openwebtext_train_file = 'data/owt_train.txt'
openwebtext_dev_file   = 'data/owt_valid.txt'
openwebtext_train_out  = 'ece496b_basics/Experiment_output/owt_train_tokens.npy'
openwebtext_dev_out    = 'ece496b_basics/Experiment_output/owt_valid_tokens.npy'

encode_and_serialize_dataset(tiny_stories_encoding, tinystories_train_file, tinystories_train_out)
encode_and_serialize_dataset(tiny_stories_encoding, tinystories_dev_file, tinystories_dev_out)

encode_and_serialize_dataset(open_webtext_encoding, openwebtext_train_file, openwebtext_train_out)
encode_and_serialize_dataset(open_webtext_encoding, openwebtext_dev_file, openwebtext_dev_out)