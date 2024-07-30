from loguru import logger
from .llama import Llama
from llmc.utils.registry_factory import MODEL_REGISTRY
from transformers import AutoConfig

try:
    from transformers import LlavaForConditionalGeneration
except:
    logger.info(
        "LlavaForConditionalGeneration is not supported in this version of transfomers. Update transfomers if you need."
    )


@MODEL_REGISTRY
class Llava(Llama):
    def __init__(self, model_path, torch_dtype):
        super().__init__(model_path, torch_dtype)

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model_config.text_config.use_cache = False
        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.llava_model.language_model
