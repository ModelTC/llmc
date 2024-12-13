import librosa
from loguru import logger
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import Qwen2AudioForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import Qwen2AudioForConditionalGeneration. '
        'If you need it, please upgrade transformers.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .qwen2 import Qwen2


@MODEL_REGISTRY
class Qwen2Audio(Qwen2):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.alm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.alm_model_config, 'use_cache'):
                self.alm_model_config.use_cache = False
        logger.info(f'self.alm_model_config : {self.alm_model_config}')
        self.alm_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.alm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.alm_model
        logger.info(f'self.alm_model : {self.alm_model}')
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.audio_model = self.alm_model.audio_tower
        self.audio_projector = self.alm_model.multi_modal_projector
        self.model = self.alm_model.language_model
        self.model_config = self.alm_model_config.text_config

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.audio_projector.linear]

    def batch_process(self, audio_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        messages = []
        answers = []
        for idx in range(len(audio_qas)):
            audio_path = audio_qas[idx]['audio']
            if audio_path is not None:
                content = []
                if not isinstance(audio_path, list):
                    audio_path = [audio_path]
                for audio_idx in range(len(audio_path)):
                    content.append({'type': 'audio', 'audio': audio_path[audio_idx]})
                if 'question' in audio_qas[idx]:
                    content.append({'type': 'text', 'text': audio_qas[idx]['question']})
                message = [{'role': 'user', 'content': content}]
            else:
                message = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': audio_qas[idx]['question']}
                        ],
                    }
                ]
            messages.append(message)
            answers.append(audio_qas[idx]['answer'] + '<|im_end|>')
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
            texts = [texts[n] + answers[n] for n in range(len(texts))]
        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{texts}')
        if not return_inputs:
            return texts
        audios = []
        for conversation in messages:
            for message in conversation:
                if isinstance(message['content'], list):
                    for ele in message['content']:
                        if ele['type'] == 'audio':
                            audios.append(
                                librosa.load(
                                    ele['audio'],
                                    sr=self.processor.feature_extractor.sampling_rate,
                                )[0]
                            )

        inputs = self.processor(
            text=texts, audios=audios, return_tensors='pt', padding=True
        ).to(next(self.alm_model.parameters()).dtype)
        return inputs
