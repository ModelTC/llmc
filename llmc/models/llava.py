from loguru import logger
from PIL import Image
from transformers import (AutoConfig, AutoProcessor,
                          LlavaForConditionalGeneration)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama


@MODEL_REGISTRY
class Llava(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

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
        self.vision_projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config

        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def batch_process(self, img_qas, calib_or_eval='eval'):
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        messages = []
        images = []
        answers = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['img']
            if img_path is not None:
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
                images.append(image)
            else:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': img_qas[idx]['question']}
                        ]
                    }
                ]
            messages.append(message)
            answers.append(img_qas[idx]['answer'])
        texts = [
            self.processor.apply_chat_template(messages[n], add_generation_prompt=True)
            for n in range(len(messages))
        ]
        if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
            texts = [
                texts[n] + ' ' + answers[n]
                for n in range(len(texts))
            ]
        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{texts}')

        inputs = self.processor(
            text=texts,
            images=images if len(images) else None,
            padding=True,
            return_tensors='pt'
        ).to(next(self.vlm_model.parameters()).dtype) # noqa
        return inputs

    def find_blocks(self, modality='language'):
        if modality == 'language':
            self.blocks = self.model.model.layers
        elif modality == 'vision':
            self.blocks = self.vision_model.vision_model.encoder.layers

    def get_vision_subsets_in_block(self, block):
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
                'do_trans': False
            },
        ]
