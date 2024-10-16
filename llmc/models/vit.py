import inspect
from collections import defaultdict

import torch
import torch.nn as nn
from loguru import logger
from transformers import (AutoConfig, ViTForImageClassification,
                          ViTImageProcessor)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


# Only verified that vit-base-patch16-224 is correct.
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

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, data_type='txt'):
        first_block_input = defaultdict(list)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                if 'output_router_logits' in kwargs:
                    assert kwargs['output_router_logits'] is False
                    kwargs.pop('output_router_logits')
                first_block_input['kwargs'].append(kwargs)
                raise ValueError

        self.move_embed_to_device('cuda')
        if data_type == 'img_txt':
            self.vision_tower = self.vision_tower.to('cuda')
            self.multi_modal_projector = self.multi_modal_projector.to('cuda')
        self.blocks[0] = self.blocks[0].cuda()
        self.blocks[0] = Catcher(self.blocks[0])

        for data in calib_data:
            try:
                if data_type == 'txt':
                    self.model(data.to(next(self.model.parameters()).device))
                elif data_type == 'img':
                    data = {
                        k: v.to(next(self.model.parameters()).device)
                        for k, v in data.items()
                    }
                    self.model(**data)
                elif data_type == 'img_txt':
                    data = {
                        k: v.to(next(self.model.parameters()).device)
                        for k, v in data.items()
                    }
                    self.vlm_model.generate(**data, max_new_tokens=200, do_sample=False)
            except ValueError:
                pass
        self.first_block_input = first_block_input
        if data_type == 'img_txt':
            self.vision_tower = self.vision_tower.cpu()
            self.multi_modal_projector = self.multi_modal_projector.cpu()
        self.blocks[0] = self.blocks[0].module
        self.blocks[0] = self.blocks[0].cpu()
        self.move_embed_to_device('cpu')
