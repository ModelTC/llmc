from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Bloom(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.transformer.h

    def find_embed_layers(self):
        self.word_embeddings = self.model.transformer.word_embeddings
        self.word_embeddings_layernorm = (
            self.model.transformer.word_embeddings_layernorm
        )

    def find_block_name(self):
        self.block_name_prefix = 'model.transformer.h'

    def get_embed_layers(self):
        return [self.word_embeddings, self.word_embeddings_layernorm]

    def get_layers_except_blocks(self):
        return [
            self.word_embeddings,
            self.word_embeddings_layernorm,
            self.model.lm_head,
            self.model.transformer.ln_f,
        ]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return True

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attention.query_key_value': (
                        block.self_attention.query_key_value
                    )
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attention.query_key_value'],
                'inspect': block.self_attention.query_key_value,
                'has_kwargs': False,
            },
            {
                'layers': {'self_attention.dense': block.self_attention.dense},
                'prev_op': [block.self_attention.query_key_value],
                'input': ['self_attention.dense'],
                'inspect': block.self_attention.dense,
                'has_kwargs': False,
            },
            {
                'layers': {'mlp.dense_h_to_4h': block.mlp.dense_h_to_4h},
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.dense_h_to_4h'],
                'inspect': block.mlp.dense_h_to_4h,
                'has_kwargs': False,
            },
            {
                'layers': {'mlp.dense_4h_to_h': block.mlp.dense_4h_to_h},
                'prev_op': [block.mlp.gelu_impl],
                'input': ['mlp.dense_4h_to_h'],
                'inspect': block.mlp.dense_4h_to_h,
                'has_kwargs': False,
            },
        ]
