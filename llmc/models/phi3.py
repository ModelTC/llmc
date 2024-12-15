from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Phi3(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.model.norm, self.model.lm_head] # noqa

    def skip_layer_name(self):
        return ['lm_head']

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
                    'self_attn.qkv_proj': block.self_attn.qkv_proj
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.qkv_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.qkv_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
                'do_trans': False
                # There is a bug. Merging scale here is not equivalence which is unreasonable.
                # Need to check! So do_trans is set to False.
            },
            {
                'layers': {'mlp.gate_up_proj': block.mlp.gate_up_proj},
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.gate_up_proj'],
                'inspect': block.mlp.gate_up_proj,
                'has_kwargs': False,
            },
            {
                'layers': {'mlp.down_proj': block.mlp.down_proj},
                'prev_op': [block.mlp.gate_up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
            },
        ]
