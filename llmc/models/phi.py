from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Phi(BaseModel):
    def __init__(self, model_path, torch_dtype, device_map=None, use_cache=False):
        super().__init__(model_path, torch_dtype, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'
        self.pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.final_layernorm]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.model.final_layernorm, self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.input_layernorm,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                    'mlp.fc1': block.mlp.fc1,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.dense': block.self_attn.dense},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.dense'],
                'inspect': block.self_attn.dense,
                'has_kwargs': False,
            },
        ]
