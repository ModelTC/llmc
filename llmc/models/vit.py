import inspect

import torch.nn as nn
from transformers import (AutoConfig, AutoProcessor, ViTForImageClassification,
                          ViTImageProcessor)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


# Only verified that vit-base-patch16-224 is correct.
@MODEL_REGISTRY
class Vit(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor = ViTImageProcessor.from_pretrained(self.model_path)
        self.model = ViTForImageClassification.from_pretrained(self.model_path)

    def find_blocks(self, modality='vision'):
        self.blocks = self.model.vit.encoder.layer

    def find_embed_layers(self):
        self.embed_tokens = self.model.vit.embeddings

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

    def get_act_fn_in_block(self, block):
        return {'intermediate.intermediate_act_fn': block.intermediate.intermediate_act_fn}

    def get_attn_in_block(self, block):
        return {'attention.attention': block.attention.attention}

    def get_matmul_in_block(self, block):
        return {
            'attention.attention.matmul_1': block.attention.attention.matmul_1,
            'attention.attention.matmul_2': block.attention.attention.matmul_2,
        }

    def get_softmax_in_block(self, block):
        return {'attention.attention.softmax': block.attention.attention.softmax}

    def __str__(self):
        return f'\nModel: \n{str(self.model)}'

    def batch_process(self, imgs):
        processor = AutoProcessor.from_pretrained(self.model_path)
        samples = []
        for img in imgs:
            sample = processor(images=img, return_tensors='pt')
            samples.append(sample)
        return samples

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attention.attention.query': block.attention.attention.query,
                    'attention.attention.key': block.attention.attention.key,
                    'attention.attention.value': block.attention.attention.value,
                },
                'prev_op': [block.layernorm_before],
                'input': ['attention.attention.query'],
                'inspect': block.attention.attention,
                'has_kwargs': True,
            },
            {
                'layers': {'attention.output.dense': block.attention.output.dense},
                'prev_op': [block.attention.attention.value],
                'input': ['attention.output.dense'],
                'inspect': block.attention.output.dense,
                'has_kwargs': False,
            },
            {
                'layers': {'intermediate.dense': block.intermediate.dense},
                'prev_op': [block.layernorm_after],
                'input': ['intermediate.dense'],
                'inspect': block.intermediate.dense,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'output.dense': block.output.dense},
                'prev_op': [block.intermediate],
                'input': ['output.dense'],
                'inspect': block.output.dense,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
