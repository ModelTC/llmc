from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Opt(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.model.decoder.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.decoder.embed_tokens
        self.embed_positions = self.model.model.decoder.embed_positions

    def find_block_name(self):
        self.block_name_prefix = 'model.decoder.layers'
        self.pairs = {'q_proj': 'qkv', 'out_proj': 'out', 'fc1': 'fc1'}

    def get_embed_layers(self):
        return [self.embed_tokens, self.embed_positions]

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.decoder.final_layer_norm]

    def get_layers_except_blocks(self):
        layers = [self.embed_tokens, self.embed_positions, self.model.lm_head]
        if self.model.model.decoder.project_in:
            layers.append(self.model.model.decoder.project_in)
        if self.model.model.decoder.project_out:
            layers.append(self.model.model.decoder.project_out)
        if self.model.model.decoder.final_layer_norm:
            layers.append(self.model.model.decoder.final_layer_norm)
        return layers

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return True

    def get_layernorms_in_block(self, block):
        return {
            'self_attn_layer_norm': block.self_attn_layer_norm,
            'final_layer_norm': block.final_layer_norm,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.self_attn_layer_norm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.out_proj': block.self_attn.out_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.out_proj'],
                'inspect': block.self_attn.out_proj,
                'has_kwargs': False,
            },
            {
                'layers': {'fc1': block.fc1},
                'prev_op': [block.final_layer_norm],
                'input': ['fc1'],
                'inspect': block.fc1,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'fc2': block.fc2},
                'prev_op': [block.fc1],
                'input': ['fc2'],
                'inspect': block.fc2,
                'has_kwargs': False,
                'is_mlp': True,
                'do_trans': False
            },
        ]
