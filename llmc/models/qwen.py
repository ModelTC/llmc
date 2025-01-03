from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Qwen(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.transformer.h

    def find_embed_layers(self):
        self.wte = self.model.transformer.wte
        self.rotary_emb = self.model.transformer.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = 'transformer.h'

    def get_embed_layers(self):
        return [self.wte, self.rotary_emb]

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.transformer.ln_f]

    def get_attn_in_block(self, block):
        return {'self_attn': block.self_attn}

    def get_layers_except_blocks(self):
        return [self.wte,
                self.rotary_emb,
                self.model.transformer.ln_f,
                self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'ln_1': block.ln_1,
            'ln_2': block.ln_2,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attn.c_attn': block.attn.c_attn
                },
                'prev_op': [block.ln_1],
                'input': ['attn.c_attn'],
                'inspect': block.attn,
                'has_kwargs': True,
            },
            {
                'layers': {'attn.c_proj': block.attn.c_proj},
                'prev_op': [block.attn.c_attn],
                'input': ['attn.c_proj'],
                'inspect': block.attn.c_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.w1': block.mlp.w1,
                    'mlp.w2': block.mlp.w2,
                },
                'prev_op': [block.ln_2],
                'input': ['mlp.w1'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.c_proj': block.mlp.c_proj},
                'prev_op': [block.mlp.w1],
                'input': ['mlp.c_proj'],
                'inspect': block.mlp.c_proj,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
