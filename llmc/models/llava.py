from loguru import logger
from PIL import Image
from transformers import AutoConfig

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama

try:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
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
        self.vision_model = self.vlm_model.vision_tower
        self.projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config
        self.need_update_mask = True

    def preprocess(self, img_qas):
        processor = AutoProcessor.from_pretrained(self.model_path)
        samples = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['img']
            txt = img_qas[idx]['question']
            txt = 'USER: <image>\n' + txt
            raw_image = Image.open(img_path)
            sample = processor(txt, raw_image, return_tensors='pt').to(next(self.vlm_model.parameters()).dtype) # noqa
            samples.append(sample)
        return samples
