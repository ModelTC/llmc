from abc import ABCMeta
from transformers import AutoTokenizer


class BaseTokenizer(metaclass=ABCMeta):
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.build_tokenizer()

    def __str__(self):
        return str(self.tokenizer)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=False, trust_remote_code=True
        )

    def get_tokenizer(self):
        return self.tokenizer
