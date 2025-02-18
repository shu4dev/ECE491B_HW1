import numpy as np
import tiktoken
from tqdm import tqdm
def encode_and_serialize_dataset(encoding, input_file, output_file):
    with open(input_file) as f:
        text = f.read()
    encoded = encoding.encode(text, allowed_special={'<|endoftext|>'})
    total_batches = 1024
    batch_size = len(encoded) // total_batches
    arr = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(len(encoded),))
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=output_file):
        batch = encoded[idx:idx+batch_size]
        arr[idx:idx+batch_size] = batch
        idx += batch_size
    arr.flush()



tiny_stories_encoding = tiktoken.get_encoding("gpt2")
open_webtext_encoding = tiktoken.get_encoding("gpt2")

#tinystories_train_file = '/home/shu4/ECE491B_HW1/data/TinyStoriesV2-GPT4-train.txt'
#tinystories_dev_file   = '/home/shu4/ECE491B_HW1/data/TinyStoriesV2-GPT4-valid.txt'
#tinystories_train_out  = '/home/shu4/ECE491B_HW1/data/Experiment_output/tinystories_train_tokens.bin'
#tinystories_dev_out    = '/home/shu4/ECE491B_HW1/data/Experiment_output/tinystories_valid_tokens.bin'

openwebtext_train_file = '/home/shu4/ECE491B_HW1/data/owt_train.txt'
openwebtext_dev_file   = '/home/shu4/ECE491B_HW1/data/owt_valid.txt'
openwebtext_train_out  = '/home/shu4/ECE491B_HW1/data/Experiment_output/owt_train_tokens.bin'
openwebtext_dev_out    = '/home/shu4/ECE491B_HW1/data/Experiment_output/owt_valid_tokens.bin'

#encode_and_serialize_dataset(tiny_stories_encoding, tinystories_train_file, tinystories_train_out)
#encode_and_serialize_dataset(tiny_stories_encoding, tinystories_dev_file, tinystories_dev_out)

encode_and_serialize_dataset(open_webtext_encoding, openwebtext_train_file, openwebtext_train_out)
encode_and_serialize_dataset(open_webtext_encoding, openwebtext_dev_file, openwebtext_dev_out)