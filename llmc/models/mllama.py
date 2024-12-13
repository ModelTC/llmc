from loguru import logger
from PIL import Image
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import MllamaForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import MllamaForConditionalGeneration. '
        'If you need it, please upgrade transformers.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama


@MODEL_REGISTRY
class Mllama(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.text_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.vision_model
        self.vision_projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config

    def batch_process(self, img_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        if len(img_qas) == 1:
            return self.single_process(img_qas[0])
        processor = AutoProcessor.from_pretrained(self.model_path)
        messages = []
        images = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['image']
            image = [Image.open(img_path)]
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
        if not return_inputs:
            return texts
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(next(self.vlm_model.parameters()).dtype) # noqa
        return inputs

    def single_process(self, img_qas):
        processor = AutoProcessor.from_pretrained(self.model_path)
        img_path = img_qas['image']
        message = [
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': img_qas['question']}]
            }
        ]
        if img_path is not None:
            image = Image.open(img_path)
            message[0]['content'].insert(0, {'type': 'image'})
        else:
            image = None
            raise NotImplementedError('Currently, only pure image-text data is supported.')
        text = processor.apply_chat_template(message, add_generation_prompt=True)
        inputs = processor(
            text=text,
            images=image,
            return_tensors='pt'
        ).to(next(self.vlm_model.parameters()).dtype)
        return inputs

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        if hasattr(block, 'cross_attn'):
            return [
                {
                    'layers': {'cross_attn.q_proj': block.cross_attn.q_proj},
                    'prev_op': [block.input_layernorm],
                    'input': ['cross_attn.q_proj'],
                    'inspect': block.cross_attn,
                    'has_kwargs': True,
                    'sub_keys': {
                        'cross_attention_states': 'cross_attention_states',
                        'attention_mask': 'cross_attention_mask',
                        'output_attentions': 'output_attentions',
                        'past_key_value': 'past_key_value',
                        'cache_position': 'cache_position',
                    }
                },
                {
                    'layers': {
                        'cross_attn.k_proj': block.cross_attn.k_proj,
                        'cross_attn.v_proj': block.cross_attn.v_proj,
                    },
                    'prev_op': [],
                    'input': ['cross_attn.k_proj'],
                    'inspect': block.cross_attn,
                    'has_kwargs': True,
                    'sub_keys': {
                        'cross_attention_states': 'cross_attention_states',
                        'attention_mask': 'cross_attention_mask',
                        'output_attentions': 'output_attentions',
                        'past_key_value': 'past_key_value',
                        'cache_position': 'cache_position',
                    }
                },
                {
                    'layers': {'cross_attn.o_proj': block.cross_attn.o_proj},
                    'prev_op': [block.cross_attn.v_proj],
                    'input': ['cross_attn.o_proj'],
                    'inspect': block.cross_attn.o_proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {
                        'mlp.gate_proj': block.mlp.gate_proj,
                        'mlp.up_proj': block.mlp.up_proj,
                    },
                    'prev_op': [block.post_attention_layernorm],
                    'input': ['mlp.gate_proj'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'mlp.down_proj': block.mlp.down_proj},
                    'prev_op': [block.mlp.up_proj],
                    'input': ['mlp.down_proj'],
                    'inspect': block.mlp.down_proj,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
            ]
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.gate_proj': block.mlp.gate_proj,
                    'mlp.up_proj': block.mlp.up_proj,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.gate_proj'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.down_proj': block.mlp.down_proj},
                'prev_op': [block.mlp.up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
