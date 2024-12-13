from typing import Optional

import librosa
import torch
from loguru import logger
from transformers import GenerationConfig

try:
    from internvl.conversation import get_conv_template
    from internvl.model.audio.processing_whisper import WhisperProcessor
    from internvl.model.internvl_chat import (InternVLChatAudioConfig,
                                              InternVLChatAudioModel)
except Exception:
    logger.warning(
        'InternOmni-internvl not installed. '
        'If you need it, please install it.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .internvl2 import load_image


def load_audio(audio_file, audio_processor):
    audio_values, _ = librosa.load(audio_file, sr=16000)

    audio_process_values = audio_processor(
        audio_values, sampling_rate=16000, return_tensors='pt'
    )
    input_features = audio_process_values['input_features']
    audio_len_after_cnn = audio_process_values['audio_len_after_cnn']
    audio_token_num = audio_process_values['audio_token_num']

    audio_input = {
        'audio_values': input_features,
        'audio_len_after_cnn': audio_len_after_cnn,
        'audio_token_num': audio_token_num,
    }
    return audio_input


@torch.no_grad()
def generate_patch_for_internvl_qwen2(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        visual_features: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_len_after_cnn: Optional[bool] = None,
        audio_token_num: Optional[bool] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    assert self.audio_context_token_id is not None

    vit_embeds = None
    if visual_features is not None:
        vit_embeds = visual_features
    elif pixel_values is not None:
        vit_embeds = self.extract_feature(pixel_values)

    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)

    if vit_embeds is not None:
        selected = (input_ids == self.img_context_token_id)
        input_embeds[selected] = vit_embeds.reshape(-1, C)

    if audio_values is not None and audio_len_after_cnn is not None and audio_token_num is not None:
        audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)
        output_audios = []
        for i in range(len(audio_token_num)):
            token_num = int(audio_token_num[i].item())
            audio = audio_embeds[i][:token_num]
            output_audios.append(audio)
        output_audios = torch.cat(output_audios, dim=0)
        selected = (input_ids == self.audio_context_token_id)
        input_embeds[selected] = output_audios.reshape(-1, C)

    input_embeds = input_embeds.reshape(B, N, C)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


@MODEL_REGISTRY
class InternOmni():
    def __new__(cls, config, device_map=None, use_cache=False):
        avlm_model_config = InternVLChatAudioConfig.from_pretrained(
            config.model.path, trust_remote_code=True
        )
        language_part = avlm_model_config.llm_config.model_type
        logger.warning(f'InternOmni language_part: {language_part}')
        if language_part == 'internlm2':
            from .internlm2 import InternLM2

            class NewClass(InternOmniSharedBehavior, InternLM2):
                def __init__(self, config, device_map=None, use_cache=False):
                    super().__init__(config, device_map, use_cache)
        elif language_part == 'qwen2':
            from .qwen2 import Qwen2

            class NewClass(InternOmniSharedBehavior, Qwen2):
                def __init__(self, config, device_map=None, use_cache=False):
                    super().__init__(config, device_map, use_cache)
                    setattr(
                        self.avlm_model,
                        'generate',
                        generate_patch_for_internvl_qwen2.__get__(self.avlm_model),
                    )
        elif language_part == 'phi3':
            from .phi3 import Phi3

            class NewClass(InternOmniSharedBehavior, Phi3):
                def __init__(self, config, device_map=None, use_cache=False):
                    super().__init__(config, device_map, use_cache)
        elif language_part == 'llama':
            from .llama import Llama

            class NewClass(InternOmniSharedBehavior, Llama):
                def __init__(self, config, device_map=None, use_cache=False):
                    super().__init__(config, device_map, use_cache)
        else:
            raise Exception(f'Not support for language_part: {language_part}')
        return NewClass(config, device_map, use_cache)


@MODEL_REGISTRY
class InternOmniSharedBehavior():
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

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
        self.mm_model = self.avlm_model
        logger.info(f'self.avlm_model : {self.avlm_model}')
        self.model = self.avlm_model.language_model
        self.model_config = self.avlm_model_config.llm_config
        if not self.use_cache:
            if hasattr(self.model_config, 'use_cache'):
                self.model_config.use_cache = False

        self.audio_model = self.avlm_model.audio_model
        self.vision_model = self.avlm_model.vision_model
        self.vision_projector = self.avlm_model.mlp1
        self.audio_projector = self.avlm_model.mlp2

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        audio_context_token_id = self.tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)

        self.avlm_model.img_context_token_id = img_context_token_id
        self.avlm_model.audio_context_token_id = audio_context_token_id
        self.avlm_model.ps_version = 'v2'

        self.audio_processor = WhisperProcessor.from_pretrained(self.model_path)

        self.default_image_prompt_template = {
            'single': '<image>\n',
            'multiple': 'Image-<|idx|>: <image>\n'
        }
        self.default_audio_prompt_template = {
            'single': '<audio>\n',
            'multiple': 'Audio-<|idx|>: <audio>\n'
        }

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.audio_projector[-1], self.vision_projector[-1]]

    def batch_process(self, audio_img_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        questions = []
        answers = []
        pixel_values_list = []
        num_audios_patches_list = []
        audio_values_list = []
        audio_len_after_cnns_list = []
        audio_token_nums_list = []

        for idx in range(len(audio_img_qas)):
            img_path = audio_img_qas[idx]['image']
            audio_path = audio_img_qas[idx]['audio']
            num_patches = []
            num_audios = []

            if audio_path is not None:
                if not isinstance(audio_path, list):
                    audio_path = [audio_path]
                for audio_idx in range(len(audio_path)):
                    audio_input = load_audio(audio_path[audio_idx], self.audio_processor)
                    audio_input['audio_values'] = audio_input['audio_values'].to(
                        next(self.avlm_model.parameters()).dtype
                    )
                    audio_values_list.append(audio_input['audio_values'])
                    num_audios.append(audio_input['audio_token_num'])
                    audio_len_after_cnns_list.append(audio_input['audio_len_after_cnn'])
                    audio_token_nums_list.append(audio_input['audio_token_num'])
                if audio_img_qas[idx]['question'].count('<audio>') == 0:
                    prefix = ''
                    if len(audio_path) == 1:
                        prefix = self.default_audio_prompt_template['single']
                    else:
                        for n in range(len(img_path)):
                            prefix = prefix + self.default_audio_prompt_template['multiple'].replace('<|idx|>', f'{n+1}') # noqa
                    audio_img_qas[idx]['question'] = prefix + audio_img_qas[idx]['question']

            if img_path is not None:
                if not isinstance(img_path, list):
                    img_path = [img_path]
                for img_idx in range(len(img_path)):
                    pixel_values = load_image(img_path[img_idx], max_num=12).to(
                        next(self.avlm_model.parameters()).dtype
                    )
                    pixel_values_list.append(pixel_values)
                    num_patches.append(pixel_values.size(0))
                if audio_img_qas[idx]['question'].count('<image>') == 0:
                    prefix = ''
                    if len(img_path) == 1:
                        prefix = self.default_image_prompt_template['single']
                    else:
                        for n in range(len(img_path)):
                            prefix = prefix + self.default_image_prompt_template['multiple'].replace('<|idx|>', f'{n+1}') # noqa
                    audio_img_qas[idx]['question'] = prefix + audio_img_qas[idx]['question']
                else:
                    assert audio_img_qas[idx]['question'].count('<image>') == len(img_path), f"{audio_img_qas[idx]['image']} this data prompt is wrong." # noqa
            num_audios_patches_list.append((num_audios, num_patches))

            questions.append(audio_img_qas[idx]['question'])
            answers.append(audio_img_qas[idx]['answer'] + '<|im_end|>')

        pixel_values = (
            torch.cat(pixel_values_list, dim=0) if len(pixel_values_list) > 0 else None
        )
        audio_values = (
            torch.cat(audio_values_list, dim=0) if len(audio_values_list) > 0 else None
        )
        generation_config = dict()

        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        AUDIO_START_TOKEN = '<audio>'
        AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'
        AUDIO_END_TOKEN = '</audio>'
        queries = []
        for idx, (num_audios, num_patches) in enumerate(num_audios_patches_list):
            question = questions[idx]
            try:
                template = get_conv_template(self.avlm_model.template)
            except Exception:
                raise Exception(
                    'InternLM2 conversation.py not be found. '
                    'Please copy it from model path to llmc/models.'
                )
            template.system_message = self.avlm_model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
                query += answers[idx]
            if calib_or_eval == 'calib':
                logger.info(f'Calib data is:\n{query}')

            for _num_audios_i in num_audios:
                audio_tokens = AUDIO_START_TOKEN + AUDIO_CONTEXT_TOKEN * _num_audios_i + AUDIO_END_TOKEN # noqa
                query = query.replace('<audio>', audio_tokens, 1)
            for _num_patches_i in num_patches:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.avlm_model.num_image_token * _num_patches_i + IMG_END_TOKEN # noqa
                query = query.replace('<image>', image_tokens, 1)

            queries.append(query)
        assert self.tokenizer.padding_side == 'left'
        if not return_inputs:
            return queries
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids(['<|im_end|>'])[0]] # noqa
        generation_config['eos_token_id'] = eos_token_id
        generation_config['temperature'] = 0
        generation_config['top_p'] = 0.1
        generation_config['top_k'] = 1
        generation_config['repetition_penalty'] = 1

        inputs = {
            'pixel_values': pixel_values,
            'audio_values': audio_values,
            'audio_len_after_cnn': torch.tensor(audio_len_after_cnns_list),
            'audio_token_num': torch.tensor(audio_token_nums_list),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            **generation_config
        }
        return inputs
