from common import  gpt2_bytes_to_unicode
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

print(b' \xe2\x80everyone, that'.decode('utf-8'))