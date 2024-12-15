from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Falcon(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.transformer.h

    def find_embed_layers(self):
        self.word_embeddings = self.model.transformer.word_embeddings
        self.rotary_emb = self.model.model.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = 'model.transformer.h'

    def get_embed_layers(self):
        return [self.word_embeddings]

    def get_attention_rotary_layers(self):
        return [self.rotary_emb]

    def get_layers_except_blocks(self):
        return [self.word_embeddings, self.rotary_emb, self.model.transformer.ln_f]

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        if block.config.architectures[0] == 'RWForCausalLM':
            new_decoder_architecture = False
        elif block.config.architectures[0] == 'FalconForCausalLM':
            new_decoder_architecture = True
        if new_decoder_architecture:
            return {'ln_attn': block.ln_attn, 'ln_mlp': block.ln_mlp}
        else:
            if block.config.parallel_attn:
                return {'input_layernorm': block.input_layernorm}
            else:
                return {'post_attention_layernorm': block.post_attention_layernorm}

    def get_subsets_in_block(self, block):
        if block.config.architectures[0] == 'RWForCausalLM':
            new_decoder_architecture = False
        elif block.config.architectures[0] == 'FalconForCausalLM':
            new_decoder_architecture = True
        if new_decoder_architecture:
            subset1 = {
                'layers': {
                    'self_attention.query_key_value': (
                        block.self_attention.query_key_value
                    )
                },
                'prev_op': [block.ln_attn],
                'input': ['self_attention.query_key_value'],
                'inspect': block.self_attention.query_key_value,
                'has_kwargs': False,
            }
            subset3 = {
                'layers': {'mlp.dense_h_to_4h': block.mlp.dense_h_to_4h},
                'prev_op': [block.ln_mlp],
                'input': ['mlp.dense_h_to_4h'],
                'inspect': block.mlp.dense_h_to_4h,
                'has_kwargs': False,
            }
        else:
            subset1 = {
                'layers': {
                    'self_attention.query_key_value': (
                        block.self_attention.query_key_value
                    )
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attention.query_key_value'],
                'inspect': block.self_attention.query_key_value,
                'has_kwargs': False,
            }
            if block.config.parallel_attn:
                subset3 = {
                    'layers': {'mlp.dense_h_to_4h': block.mlp.dense_h_to_4h},
                    'prev_op': [block.input_layernorm],
                    'input': ['mlp.dense_h_to_4h'],
                    'inspect': block.mlp.dense_h_to_4h,
                    'has_kwargs': False,
                }
            else:
                subset3 = {
                    'layers': {'mlp.dense_h_to_4h': block.mlp.dense_h_to_4h},
                    'prev_op': [block.post_attention_layernorm],
                    'input': ['mlp.dense_h_to_4h'],
                    'inspect': block.mlp.dense_h_to_4h,
                    'has_kwargs': False,
                }

        subset2 = {
            'layers': {'self_attention.dense': block.self_attention.dense},
            'prev_op': [block.self_attention.query_key_value],
            'input': ['self_attention.dense'],
            'inspect': block.self_attention.dense,
            'has_kwargs': False,
        }
        subset4 = {
            'layers': {'mlp.dense_4h_to_h': block.mlp.dense_4h_to_h},
            'prev_op': [block.mlp.act],
            'input': ['mlp.dense_4h_to_h'],
            'inspect': block.mlp.dense_4h_to_h,
            'has_kwargs': False,
        }
        return [subset1, subset2, subset3, subset4]
