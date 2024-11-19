from loguru import logger
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import Qwen2VLForConditionalGeneration. '
        'Please upgrade transformers.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .qwen2 import Qwen2


@MODEL_REGISTRY
class Qwen2VL(Qwen2):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )

        self.vision_model = self.vlm_model.visual
        self.projector = self.vision_model.merger
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config

    def batch_process(self, img_qas):
        from qwen_vl_utils import process_vision_info
        processor = AutoProcessor.from_pretrained(self.model_path)
        messages = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['img']
            if img_path is not None:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image', 'image': img_path,
                                'resized_height': 280, 'resized_width': 420
                                # default: original resolution
                            },
                            {'type': 'text', 'text': img_qas[idx]['question']}
                        ]
                    }
                ]
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
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors='pt',
        ).to(next(self.vlm_model.parameters()).dtype)
        return inputs
