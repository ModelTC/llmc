import inspect

import torch.nn as nn

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class ChatGLM(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.transformer.encoder.layers

    def find_embed_layers(self):
        self.embedding = self.model.transformer.embedding
        self.rotary_pos_emb = self.model.transformer.rotary_pos_emb

    def find_block_name(self):
        self.block_name_prefix = 'transformer.encoder.layers'

    def get_embed_layers(self):
        return [self.embedding]

    def get_attention_rotary_layers(self):
        return [self.rotary_pos_emb]

    def get_head_layers(self):
        return [self.model.transformer.output_layer]

    def get_pre_head_layernorm_layers(self):
        return [self.model.transformer.encoder.final_layernorm]

    def get_layers_except_blocks(self):
        return [self.embedding, self.rotary_pos_emb, self.model.transformer.output_layer, self.model.transformer.encoder.final_layernorm] # noqa

    def skip_layer_name(self):
        return ['final_layernorm']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attention.query_key_value': block.self_attention.query_key_value
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attention.query_key_value'],
                'inspect': block.self_attention,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attention.dense': block.self_attention.dense},
                'prev_op': [block.self_attention.query_key_value],
                'input': ['self_attention.dense'],
                'inspect': block.self_attention.dense,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.dense_h_to_4h': block.mlp.dense_h_to_4h
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.dense_h_to_4h'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.down_proj': block.mlp.dense_4h_to_h},
                'prev_op': [block.mlp.dense_h_to_4h],
                'input': ['mlp.dense_4h_to_h'],
                'inspect': block.mlp.dense_4h_to_h,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
