from loguru import logger
from transformers import (AutoConfig, ViTForImageClassification,
                          ViTImageProcessor)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Vit(BaseModel):
    def __init__(self, model_path, torch_dtype, device_map=None, use_cache=False):
        super().__init__(model_path, torch_dtype, device_map, use_cache)

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor = ViTImageProcessor.from_pretrained(self.model_path)
        self.model = ViTForImageClassification.from_pretrained(self.model_path)

    def find_blocks(self):
        self.blocks = self.model.vit.encoder.layer

    def find_embed_layers(self):
        self.embed_tokens = self.model.vit.embeddings

    def find_block_name(self):
        self.block_name_prefix = 'encoder.layer'
        self.pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_head_layers(self):
        return ['classifier']

    def get_pre_head_layernorm_layers(self):
        return [self.model.layernorm]

    def get_layers_except_blocks(self):
        return [self.embed_tokens, self.model.layernorm, self.model.classifier]

    def skip_layer_name(self):
        return ['classifier']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'layernorm_before': block.layernorm_before,
            'layernorm_after': block.layernorm_after,
        }

    def __str__(self):
        return f'\nModel: \n{str(self.model)}'

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attention.query': block.attention.query,
                    'attention.key': block.attention.key,
                    'attention.value': block.attention.value,
                },
                'prev_op': [block.layernorm_before],
                'input': ['attention.query'],
                'inspect': block.attention,
                'has_kwargs': True,
            },
            {
                'layers': {'attention.output.dense': block.attention.output.dense},
                'prev_op': [block.attention.value],
                'input': ['attention.output'],
                'inspect': block.attention.output,
                'has_kwargs': False,
            },
            {
                'layers': {'intermediate.dense': block.intermediate.dense},
                'prev_op': [block.layernorm_after],
                'input': ['intermediate.dense'],
                'inspect': block.intermediate,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'output.dense': block.output.dense},
                'prev_op': [block.intermediate],
                'input': ['output.dense'],
                'inspect': block.output,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
