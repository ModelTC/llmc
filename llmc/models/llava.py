import inspect

import torch.nn as nn
from loguru import logger
from PIL import Image
from transformers import (AutoConfig, AutoProcessor,
                          LlavaForConditionalGeneration)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama


@MODEL_REGISTRY
class Llava(Llama):
    def __init__(self, model_path, torch_dtype, device_map=None, use_cache=False):
        super().__init__(model_path, torch_dtype, device_map, use_cache)
        self.is_vlm = True

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.text_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.vision_model = self.vlm_model.vision_tower
        self.projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config

    def find_encoder_blocks(self):
        self.encoder_blocks = self.vision_model.vision_model.encoder.layers

    def get_encoder_catcher(self, first_block_input):

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

        return Catcher

    def batch_process(self, img_qas):
        if len(img_qas) == 1:
            return self.single_process(img_qas[0])
        processor = AutoProcessor.from_pretrained(self.model_path)
        messages = []
        images = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['img']
            image = Image.open(img_path)
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {'type': 'text', 'text': img_qas[idx]['question']}
                    ]
                }
            ]
            messages.append(message)
            images.append(image)
        texts = [
            processor.apply_chat_template(msg, add_generation_prompt=True)
            for msg in messages
        ]
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(next(self.vlm_model.parameters()).dtype) # noqa
        return inputs

    def single_process(self, img_qas):
        processor = AutoProcessor.from_pretrained(self.model_path)
        img_path = img_qas['img']
        image = Image.open(img_path) if img_path is not None else None
        message = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': img_qas['question']}]
            }
        ]
        if img_path is not None:
            message[0]['content'].insert(0, {'type': 'image'})
        text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(
            text=text,
            images=image,
            padding=True,
            return_tensors='pt'
        ).to(next(self.vlm_model.parameters()).dtype) # noqa
        return inputs

    def get_encoder_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.layer_norm1],
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
                'layers': {'mlp.fc1': block.mlp.fc1},
                'prev_op': [block.layer_norm2],
                'input': ['mlp.fc1'],
                'inspect': block.mlp.fc1,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.fc2': block.mlp.fc2},
                'prev_op': [block.mlp.fc1],
                'input': ['mlp.fc2'],
                'inspect': block.mlp.fc2,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
