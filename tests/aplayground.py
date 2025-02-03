#!/usr/bin/env python3
from __future__ import annotations
from typing import IO, BinaryIO, Iterable, Iterator, Optional, Type, Tuple, List, Dict
import json
import regex as re
import tiktoken
from common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: Optional[list[str]] = None,
):
    return Tokenizer(vocab, merges, special_tokens)

class Tokenizer:

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merge = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []
        
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        gpt2_bpe_merges = []
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)

        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }

        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token
        
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        ids = []
        tokens = []

        def escape_token(token):
            escaped = re.escape(token)
            return f"{escaped}"

        special_tokens_pattern = "|".join(f"(?:{escape_token(token)})" for token in special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|[^\S\n]+(?!\S)|[^\S\n]+"""
        base_pattern = (
            r"[^\S\n]*[\p{L}\p{N}]+|"      # Optional leading whitespace + words
            r"\s*\p{So}|"             # Optional leading whitespace + emojis
            r"\s*'(?:[sdmt]|ll|ve|re)|" # Optional leading whitespace + contractions
            r"[^\s\p{L}\p{N}\p{So}]|" # Optional leading whitespace + other characters
            r"\s+"                     # Any remaining whitespace
        )

        final_pattern = f"(?:{special_tokens_pattern})|{base_pattern}" if special_tokens else PAT
        tokens = re.findall(final_pattern, text)
        print(tokens)
        for token in tokens:
            
            if token in self.special_tokens:
                ids.append(self.token_to_id[token.encode("utf-8")])
                continue
            else:
                token = [bytes([b]) for b in token.encode("utf-8")]
            do_merge = True
            while do_merge:
                best_idx = 0
                best_score = len(self.merge)
                for idx in range(len(token) - 1):
                    pair = (token[idx], token[idx + 1])
                    if pair in self.merge:
                        score = self.merge.index(pair)
                        if score <= best_score:
                            best_score = score
                            best_idx = idx
                if best_score == len(self.merge):
                    for b in token:
                        ids.append(self.token_to_id[b])
                    do_merge = False
                if do_merge:
                    token[best_idx] = token[best_idx] + token[best_idx + 1]
                    del token[best_idx + 1]
        return ids  
                    
          
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids


    def decode(self, ids: List[int]) -> str:
        tokens = b""
        for i in ids:
            tokens += self.id_to_token[i]
        return tokens.decode("utf-8", errors="replace")

def test_roundtrip_unicode_string_with_special_tokens():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(
        corpus_contents, allowed_special={"<|endoftext|>"}
    )
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids
    print(reference_ids)
    print(ids)
    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents

test_roundtrip_unicode_string_with_special_tokens()