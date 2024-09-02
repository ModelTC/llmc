from loguru import logger
from transformers import AutoConfig

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama

try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    logger.info(
        'LlavaForConditionalGeneration is not supported in this version of transfomers.'
        'Update transfomers if you need.'
    )


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
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config
