from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Starcoder(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.transformer.h

    def find_embed_layers(self):
        self.embed_tokens_1 = self.model.transformer.wte
        self.embed_tokens_2 = self.model.transformer.wpe

    def find_block_name(self):
        self.block_name_prefix = 'model.transformer.h'

    def get_embed_layers(self):
        return [self.embed_tokens_1, self.embed_tokens_2]

    def get_layers_except_blocks(self):
        return [
            self.embed_tokens_1,
            self.embed_tokens_2,
            self.model.transformer.ln_f,
            self.model.lm_head,
        ]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return True

    def get_layernorms_in_block(self, block):
        return {
            'ln_1': block.ln_1,
            'ln_2': block.ln_2,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attn.c_attn': block.attn.c_attn,
                },
                'prev_op': [block.ln_1],
                'input': ['attn.c_attn'],
                'inspect': block.attn.c_attn,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'attn.c_proj': block.attn.c_proj,
                },
                'prev_op': [block.attn.c_attn],
                'input': ['attn.c_proj'],
                'inspect': block.attn.c_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.c_fc': block.mlp.c_fc,
                },
                'prev_op': [block.ln_2],
                'input': ['mlp.c_fc'],
                'inspect': block.mlp.c_fc,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.c_proj': block.mlp.c_proj,
                },
                'prev_op': [block.mlp.c_fc],
                'input': ['mlp.c_proj'],
                'inspect': block.mlp.c_proj,
                'has_kwargs': False,
            },
        ]
