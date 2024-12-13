from loguru import logger
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

from llmc.utils.registry_factory import MODEL_REGISTRY

from .chatglm import ChatGLM


@MODEL_REGISTRY
class GLM4V(ChatGLM):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.transformer.vision
        self.vision_projector = self.vlm_model.transformer.vision.linear_proj
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.vision_projector.dense_4h_to_h]

    def batch_process(self, img_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        assert return_inputs, 'return_inputs should be True for GLM4V.'
        messages = []
        answers = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['image']
            if img_path is not None:
                image = Image.open(img_path).convert('RGB')
                message = [
                    {
                        'role': 'user',
                        'image': image,
                        'content': img_qas[idx]['question'],
                    }
                ]
            else:
                message = [{'role': 'user', 'content': img_qas[idx]['question']}]
            messages.append(message)
            answers.append(img_qas[idx]['answer'])
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors='pt',
            return_dict=True,
            padding=True,
        )
        if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
            raise Exception(
                'glm4v not support add_answer. '
                'Maybe you can modify tokenization_chatglm.py in model path.'
            )
        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{inputs}')

        inputs = inputs.to(next(self.vlm_model.parameters()).dtype)
        return inputs
