from __future__ import annotations
from typing import  Iterable, Optional, Dict, List, Tuple, Iterator
import regex as re
import json

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
        SPAT = r"""'(?:[sdmt]|ll|ve|re|)| ?\p{L}+| ?\p{N}+| ?\p{So}+| ?[^\s\p{L}\p{N}\p{S}]+|\s+(?!\S)|\s+"""
        PAT = r"""'(?:[sdmt]|ll|ve|re|)| ?\p{L}+| ?\p{N}+| ?\p{So}+| ?\p{S}+| ?[^\s\p{L}\p{N}\p{S}]+|\s+(?!\S)|\s+"""

        final_pattern = f"(?:{special_tokens_pattern})|{SPAT}" if special_tokens else PAT
        tokens = re.findall(final_pattern, text)
        
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

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d