import inspect
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image
from loguru import logger
from PIL import Image
from safetensors import safe_open
from transformers import CLIPVisionModel

from llmc.compression.quantization.module_utils import LlmcWanTransformerBlock
from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY

from .wan_t2v import WanT2V


@MODEL_REGISTRY
class WanI2V(WanT2V):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        image_encoder = CLIPVisionModel.from_pretrained(
            self.model_path, subfolder='image_encoder', torch_dtype=torch.float32
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.model_path, subfolder='vae', torch_dtype=torch.float32
        )
        self.Pipeline = WanImageToVideoPipeline.from_pretrained(
            self.model_path,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
        )
        self.find_llmc_model()
        self.find_blocks()
        for block_idx, block in enumerate(self.blocks):
            new_block = LlmcWanTransformerBlock.new(block)
            self.Pipeline.transformer.blocks[block_idx] = new_block
        self.lora_path = self.config.model.get('lora_path', None)
        if self.lora_path is not None:
            logger.info('Loading lora weights...')
            self.load_lora_weights()

        logger.info(f'self.model : {self.model}')

    def pre_process(self, image_path):
        image = load_image(image_path)
        max_area = self.target_height * self.target_width
        aspect_ratio = image.height / image.width
        mod_value = (
            self.Pipeline.vae_scale_factor_spatial * self.model.config.patch_size[1]
        )
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, width, height

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)
        Catcher = self.get_catcher(first_block_input)
        self.blocks[0] = Catcher(self.blocks[0])
        self.Pipeline.to('cuda')
        for data in calib_data:
            self.blocks[0].step = 0
            try:
                image, width, height = self.pre_process(data['image'])
                self.Pipeline(
                    image=image,
                    prompt=data['prompt'],
                    negative_prompt=data['negative_prompt'],
                    height=height,
                    width=width,
                    num_frames=self.num_frames,
                    guidance_scale=self.guidance_scale,
                )
            except ValueError:
                pass

        self.first_block_input = first_block_input
        assert len(self.first_block_input['data']) > 0, 'Catch input data failed.'
        self.n_samples = len(self.first_block_input['data'])
        logger.info(f'Retrieved {self.n_samples} calibration samples for T2V.')
        self.blocks[0] = self.blocks[0].module
        self.Pipeline.to('cpu')

    def load_lora_weights(self, alpha=1.0):
        state_dict = self.model.state_dict()
        model_index_file = os.path.join(
            self.lora_path, 'diffusion_pytorch_model.safetensors.index.json'
        )

        with open(model_index_file, 'r') as f:
            model_index = json.load(f)

        weight_map = model_index['weight_map']
        model_keys = list(state_dict.keys())

        matched_keys = {}
        for model_key in model_keys:
            if not model_key.endswith('.weight'):
                continue
            base_name = model_key.replace('.weight', '')
            for lora_key in weight_map:
                if base_name in lora_key:
                    if model_key not in matched_keys:
                        matched_keys[model_key] = []
                    matched_keys[model_key].append(lora_key)

        for weight_name in matched_keys:
            lora_A_name, lora_B_name = matched_keys[weight_name]
            weight = state_dict[weight_name].cuda()
            lora_A_path = os.path.join(self.lora_path, weight_map[lora_A_name])
            with safe_open(lora_A_path, framework='pt', device='cuda') as f:
                lora_A_weight = f.get_tensor(lora_A_name)

            lora_B_path = os.path.join(self.lora_path, weight_map[lora_B_name])
            with safe_open(lora_B_path, framework='pt', device='cuda') as f:
                lora_B_weight = f.get_tensor(lora_B_name)

            merge_weight = weight + (lora_B_weight @ lora_A_weight) * alpha
            state_dict[weight_name] = merge_weight.cpu()

        self.model.load_state_dict(state_dict)
