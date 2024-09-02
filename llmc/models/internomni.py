from loguru import logger

try:
    from internvl.model.internvl_chat import (InternVLChatAudioConfig,
                                              InternVLChatAudioModel)
except Exception:
    logger.info(
        'InternOmni-internvl not installed. '
        'If you need it, please install it.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .internlm2 import InternLM2


@MODEL_REGISTRY
class InternOmni(InternLM2):
    def __init__(self, model_path, torch_dtype, device_map=None, use_cache=False):
        super().__init__(model_path, torch_dtype, device_map, use_cache)

    def build_model(self):
        self.avlm_model_config = InternVLChatAudioConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logger.info(f'self.avlm_model_config : {self.avlm_model_config}')
        self.avlm_model = InternVLChatAudioModel.from_pretrained(
            self.model_path,
            config=self.avlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.avlm_model.language_model
        self.model_config = self.avlm_model_config.llm_config
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False
