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
