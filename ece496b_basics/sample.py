import random
import numpy as np
import time
from tokenizer import Tokenizer
def sampling(Tokenizer, file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file if line.strip()]
    sampled_documents = random.sample(documents, 10)
    byte_throughputs = []  
    
    for idx, doc in enumerate(sampled_documents, start=1):
        start_time = time.perf_counter()
        original_size = len(doc.encode('utf-8'))
        token_ids = Tokenizer.encode(doc)
        token_count = len(token_ids)
        token_array = np.array(token_ids, dtype=np.uint16)
        encoded_size = token_array.nbytes  
        avg_original_bytes_per_token = original_size / token_count if token_count > 0 else float('inf')
        compression_factor = original_size / encoded_size if encoded_size > 0 else float('inf')
        processing_time = time.perf_counter() - start_time
        byte_throughput = original_size / processing_time if processing_time > 0 else 0
        byte_throughputs.append(byte_throughput)
        
        print(f"Document {idx}:")
        print(f"  Original size: {original_size} bytes")
        print(f"  Token count: {token_count}")
        print(f"  Encoded size: {encoded_size} bytes (2 bytes per token)")
        print(f"  Average original bytes per token: {avg_original_bytes_per_token:.2f} bytes/token")
        print(f"  Compression factor: {compression_factor:.2f} (original size / encoded size)")
        print(f"  Processing time: {processing_time:.6f} seconds")
        print(f"  Throughput: {byte_throughput:.2f} bytes/second\n")
    
    average_throughput = sum(byte_throughputs) / len(byte_throughputs)
    print(f"Average throughput: {average_throughput:.2f} bytes/second")
    
if __name__ == '__main__':

    tinystories_tokenizer = Tokenizer.from_files(
        vocab_filepath="ece496b_basics/Experiment_output/tinystories_vocab.json",
        merges_filepath= "ece496b_basics/Experiment_output/tinystories_merges.txt",
    )

    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="ece496b_basics/Experiment_output/HF.owt_vocab.json",
        merges_filepath= "ece496b_basics/Experiment_output/HF.owt_merges_txt",
    )

    print("Tinystories Tokenizer\n")
    sampling(
        Tokenizer=tinystories_tokenizer,
        file_path="data/TinyStoriesV2-GPT4-train.txt"
    )

    print("-------------------------------------------------------")
    print("owt tokenizer")
    sampling(
        Tokenizer=owt_tokenizer,
        file_path="data/owt_train.txt"
    )

    print("-------------------------------------------------------")
    print("tinystores tokenizer with owt text")
    sampling(
        Tokenizer=tinystories_tokenizer,
        file_path="data/owt_train.txt"
    )