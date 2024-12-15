from typing import List, Tuple

from llmc.compression.quantization.module_utils import _TRANSFORMERS_LN_TYPES_
from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class InternLM2(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        global _TRANSFORMERS_LN_TYPES_
        _TRANSFORMERS_LN_TYPES_ += [type(self.model.model.norm)]

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.tok_embeddings = self.model.model.tok_embeddings

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'

    def get_embed_layers(self):
        return [self.tok_embeddings]

    def get_head_layers(self):
        return [self.model.output]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_layers_except_blocks(self):
        return [self.tok_embeddings, self.model.model.norm, self.model.output]

    def get_attn_in_block(self, block):
        return {'attention': block.attention}

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'attention_norm': block.attention_norm,
            'ffn_norm': block.ffn_norm,
        }

    # flake8: noqa
    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction=''):
        if history is None:
            history = []
        if tokenizer.add_bos_token:
            prompt = ''
        else:
            prompt = tokenizer.bos_token
        if meta_instruction:
            prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
        for record in history:
            prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
        prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
        return prompt

    # flake8: noqa
    def apply_chat_template(self, prompt):
        meta_instruction = 'You are an AI assistant whose name is InternLM (书生·浦语).\n'
        '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory '
        '(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n'
        '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such '
        'as English and 中文.'
        text = self.build_inputs(self.tokenizer, prompt, history=[], meta_instruction=meta_instruction)
        return text

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {'attention.wqkv': block.attention.wqkv},
                'prev_op': [block.attention_norm],
                'input': ['attention.wqkv'],
                'inspect': block.attention,
                'has_kwargs': True,
            },
            {
                'layers': {'attention.wo': block.attention.wo},
                'prev_op': [block.attention.wqkv],
                'input': ['attention.wo'],
                'inspect': block.attention.wo,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'feed_forward.w3': block.feed_forward.w3,
                    'feed_forward.w1': block.feed_forward.w1,
                },
                'prev_op': [block.ffn_norm],
                'input': ['feed_forward.w1'],
                'inspect': block.feed_forward,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'feed_forward.w2': block.feed_forward.w2},
                'prev_op': [block.feed_forward.w3],
                'input': ['feed_forward.w2'],
                'inspect': block.feed_forward.w2,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
