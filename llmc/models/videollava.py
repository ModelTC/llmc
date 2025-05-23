from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from lmms_eval.api.model import lmms
from lmms_eval.models.video_llava import VideoLLaVA as VL
from loguru import logger
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, VideoLlavaForConditionalGeneration,
                          VideoLlavaProcessor)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama


@MODEL_REGISTRY
class VideoLLaVA(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.text_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = VideoLlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.eval_name = 'VideoLLaVAHfEval'
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.video_tower = self.vlm_model.video_tower
        self.image_tower = self.vlm_model.image_tower
        self.vision_projector = self.vlm_model.multi_modal_projector
        self.model = self.vlm_model.language_model
        self.model_config = self.vlm_model_config.text_config
        self.pruning_config = {
            'is_video_model': True,
            'image_token_length': self.vlm_model_config.image_seq_length,
            'video_token_length': self.vlm_model_config.video_seq_length,
            'select_layer': self.vlm_model_config.vision_feature_layer,
            'select_feature': self.vlm_model_config.vision_feature_select_strategy,
            'image_token_index': self.vlm_model_config.image_token_index,
            'video_token_index': self.vlm_model_config.video_token_index,
        }


@MODEL_REGISTRY
class VideoLLaVAHfEval(VL):
    def __init__(
        self,
        llmc_model,
        pretrained: str = 'LanguageBind/Video-LLaVA-7B-hf',
        truncation: Optional[bool] = True,
        device: Optional[str] = 'cuda:0',
        dtype: Optional[Union[str, torch.dtype]] = 'auto',
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        attn_implementation=(
            'sdpa' if torch.__version__ > '2.1.2' else 'eager'
        ),
        # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2".
        # Seems FA2 is not effective during inference:
        # https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map='cuda:0',
        conv_template='llava_v1',
        use_cache=True,
        truncate_context=False,
        num_frames: int = 8,
        # whether to truncate the context in generation,
        # set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        elif accelerator.num_processes == 1 and device_map == 'auto':
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'

        self.pretrained = pretrained
        self._model = llmc_model.cuda()
        self._processor = VideoLlavaProcessor.from_pretrained(pretrained)
        self.prompt = 'USER: <video>{}? ASSISTANT:'
        self.num_frames = num_frames
        assert (
            num_frames == 8
        ), 'num_frames must be 8'
        # self.model_name = get_model_name_from_path(pretrained)
        # self._tokenizer, self._model, self.processor,
        # self._max_length = load_pretrained_model(pretrained,
        # None, self.model_name, device_map=self.device_map)
        # self.video_processor = self.processor["video"]
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1,
        # "Llava currently does not support batched generation.
        # See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], 'Unsupported distributed type provided. Only DDP and FSDP are supported.'
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    'train_micro_batch_size_per_gpu': self.batch_size_per_gpu,
                    'train_batch_size': self.batch_size_per_gpu
                    * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                logger.info(
                    'Detected that you are using DistributedType.DEEPSPEED. ' +
                    'Make sure you run `accelerate config` and set zero stage to 0'
                )
            if (
                accelerator.distributed_type == DistributedType.FSDP
                or accelerator.distributed_type == DistributedType.DEEPSPEED
            ):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                logger.info(
                    f'Using {accelerator.num_processes} devices with data parallelism'
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == 'auto':
            logger.info(
                f'Using {accelerator.num_processes} devices with tensor parallelism'
            )
            self._rank = 0
            self._world_size = 1
        else:
            logger.info(f'Using single device: {self._device}')
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
