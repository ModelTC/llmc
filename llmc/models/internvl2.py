from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

from llmc.utils.registry_factory import MODEL_REGISTRY

from .internlm2 import InternLM2


@MODEL_REGISTRY
class InternVL2(InternLM2):
    def __init__(self, model_path, torch_dtype):
        super().__init__(model_path, torch_dtype)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if hasattr(self.vlm_model_config, 'use_cache'):
            self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.llm_config
