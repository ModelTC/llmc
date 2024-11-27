import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmc.utils.registry_factory import MODEL_REGISTRY

from .internlm2 import InternLM2

try:
    from .conversation import get_conv_template
except Exception:
    logger.warning(
        'InternLM2 conversation.py not be found. '
        'If you need it, please copy it from model path to llmc/models.'
    )


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):  # noqa
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if # noqa
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num) # noqa
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@MODEL_REGISTRY
class InternVL2(InternLM2):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.vlm_model.language_model
        self.vision_model = self.vlm_model.vision_model
        self.vision_projector = self.vlm_model.mlp1
        self.model_config = self.vlm_model_config.llm_config
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.vlm_model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN) # noqa

        self.default_image_prompt_template = {
            'single': '<image>\n',
            'multiple': 'Image-<|idx|>: <image>\n'
        }
        logger.warning(
            f'InternVL2 default_image_prompt_template: {self.default_image_prompt_template}'
        )
        logger.warning(
            'Default template refer to the link https://huggingface.co/OpenGVLab/InternVL2-2B. '
            'If you want a custom template, you can change it. '
            'Besides, you can also put the <image> into your calib dataset.'
        )

    def batch_process(self, img_qas, calib_or_eval='eval'):
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        questions = []
        answers = []
        pixel_values_list = []
        num_patches_list = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['img']
            num_patches = []
            if img_path is not None:
                if not isinstance(img_path, list):
                    img_path = [img_path]
                for img_idx in range(len(img_path)):
                    pixel_values = load_image(img_path[img_idx], max_num=12).to(
                        next(self.vlm_model.parameters()).dtype
                    )
                    pixel_values_list.append(pixel_values)
                    num_patches.append(pixel_values.size(0))
            num_patches_list.append(num_patches)
            if img_path is not None:
                if img_qas[idx]['question'].count('<image>') == 0:
                    prefix = ''
                    if len(img_path) == 1:
                        prefix = self.default_image_prompt_template['single']
                    else:
                        for n in range(len(img_path)):
                            prefix = prefix + self.default_image_prompt_template['multiple'].replace('<|idx|>', f'{n+1}') # noqa
                    img_qas[idx]['question'] = prefix + img_qas[idx]['question']
                else:
                    assert img_qas[idx]['question'].count('<image>') == len(img_path), f"{img_qas[idx]['img']} this data prompt is wrong." # noqa
            questions.append(img_qas[idx]['question'])
            answers.append(img_qas[idx]['answer'] + '<|im_end|>')

        pixel_values = (
            torch.cat(pixel_values_list, dim=0) if len(pixel_values_list) > 0 else None
        )
        generation_config = dict()

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            try:
                template = get_conv_template(self.vlm_model.template)
            except Exception:
                raise Exception(
                    'InternLM2 conversation.py not be found. '
                    'Please copy it from model path to llmc/models.'
                )
            template.system_message = self.vlm_model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
                query += answers[idx]
            if calib_or_eval == 'calib':
                logger.info(f'Calib data is:\n{query}')
            for _num_patches_i in num_patches:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.vlm_model.num_image_token * _num_patches_i + IMG_END_TOKEN # noqa
                query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        assert self.tokenizer.padding_side == 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id

        inputs = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            **generation_config
        }
        return inputs

    def find_blocks(self, modality='language'):
        if modality == 'language':
            self.blocks = self.model.model.layers
        elif modality == 'vision':
            self.blocks = self.vision_model.encoder.layers

    def get_vision_subsets_in_block(self, block):
        return [
            {
                'layers': {'attn.qkv': block.attn.qkv},
                'prev_op': [block.norm1],
                'input': ['attn.qkv'],
                'inspect': block.attn,
                'has_kwargs': False,
            },
            {
                'layers': {'attn.proj': block.attn.proj},
                'prev_op': [block.attn.qkv],
                'input': ['attn.proj'],
                'inspect': block.attn.proj,
                'has_kwargs': False,
            },
            {
                'layers': {'mlp.fc1': block.mlp.fc1},
                'prev_op': [block.norm2],
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
