from abc import ABCMeta

from transformers import AutoConfig, AutoTokenizer


class BaseTokenizer(metaclass=ABCMeta):
    def __init__(self, tokenizer_path, tokenizer_mode, model_type):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_mode = tokenizer_mode
        self.model_type = model_type
        if self.tokenizer_mode == 'fast':
            self.use_fast = True
        else:
            self.use_fast = False
        if self.model_type == 'Vit':
            self.tokenizer = None
        else:
            self.build_tokenizer()
            self.patch()

    def __str__(self):
        return str(self.tokenizer)

    def build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, use_fast=self.use_fast, trust_remote_code=True
        )

    def get_tokenizer(self):
        return self.tokenizer

    def patch(self):
        if 'Intern' in self.model_type:
            self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
