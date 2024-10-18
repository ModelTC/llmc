from abc import ABCMeta

from transformers import AutoTokenizer


class BaseTokenizer(metaclass=ABCMeta):
    def __init__(self, tokenizer_path, tokenizer_mode):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_mode = tokenizer_mode
        if self.tokenizer_mode == 'fast':
            self.use_fast = True
        else:
            self.use_fast = False
        self.build_tokenizer()

    def __str__(self):
        return str(self.tokenizer)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=self.use_fast, trust_remote_code=True
        )

    def get_tokenizer(self):
        return self.tokenizer
