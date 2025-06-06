from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from lmms_eval.api.model import lmms
from lmms_eval.models.llava_onevision import Llava_OneVision as LLaVA_OV
from loguru import logger
from packaging import version
from transformers import AutoConfig

from llmc.utils.registry_factory import MODEL_REGISTRY

from .llama import Llama

try:
    from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                 DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                 IMAGE_TOKEN_INDEX)
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (KeywordsStoppingCriteria,
                                get_model_name_from_path, process_images,
                                tokenizer_image_token)
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    logger.debug(
        f'LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}'
    )

# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse('2.1.2'):
    best_fit_attn_implementation = 'sdpa'
else:
    best_fit_attn_implementation = 'eager'


@MODEL_REGISTRY
class Llava_OneVision(Llama):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            self.vlm_model_config.text_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')

        llava_model_args = {
            'multimodal': True,
        }
        llava_model_args['attn_implementation'] = best_fit_attn_implementation

        model_name = 'llava_qwen'

        overwrite_config = {}
        overwrite_config['mm_spatial_pool_stride'] = 2
        overwrite_config['mm_spatial_pool_mode'] = 'bilinear'

        llava_model_args['overwrite_config'] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self.tokenizer, self.vlm_model, image_processor, max_length = (
                load_pretrained_model(
                    self.model_path,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop('multimodal', None)
            self.tokenizer, self.vlm_model, image_processor, max_length = (
                load_pretrained_model(
                    self.model_path,
                    None,
                    model_name,
                    device_map=self.device_map,
                    **llava_model_args,
                )
            )

        self.vlm_model.image_processor = image_processor
        self.vlm_model.max_length = max_length
        self.vlm_model.tokenizer = self.tokenizer

        self.eval_name = 'Llava_OneVision_Eval'
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.get_vision_tower()
        self.vision_projector = self.vlm_model.model.mm_projector
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config.text_config
        self.pruning_config = {
            'is_video_model': False,
            'image_token_length': self.vlm_model_config.image_seq_length,
            'select_layer': self.vlm_model_config.vision_feature_layer,
            'select_feature': self.vlm_model_config.vision_feature_select_strategy,
            'image_token_index': self.vlm_model_config.image_token_index,
        }

        self.processor = None

    def find_blocks(self):
        if self.get_modality() == 'language':
            super().find_blocks()
        elif self.get_modality() == 'vision':
            self.blocks = self.vision_model.vision_tower.vision_model.encoder.layers
        else:
            raise Exception(f'Llava_OneVision do not support {self.get_modality()} modality.')


@MODEL_REGISTRY
class Llava_OneVision_Eval(LLaVA_OV):
    """Llava Model."""

    def __init__(
        self,
        llmc_model,
        pretrained: str = 'liuhaotian/llava-v1.5-7b',
        truncation: Optional[bool] = True,
        device: Optional[str] = 'cuda:0',
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = 'cuda:0',
        conv_template: Optional[str] = 'qwen_1_5',
        use_cache: Optional[bool] = True,
        truncate_context: Optional[
            bool
        ] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        mm_spatial_pool_stride: Optional[int] = 2,
        mm_spatial_pool_mode: Optional[str] = 'bilinear',
        token_strategy: Optional[
            str
        ] = 'single',  # could be "single" or "multiple", "multiple"
        # denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = 'decord',
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        # Do not use kwargs for now
        assert kwargs == {}, f'Unexpected kwargs: {kwargs}'

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
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num
        self.mm_spatial_pool_stride = mm_spatial_pool_stride
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        self.video_decode_backend = video_decode_backend

        # cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        self._model = llmc_model.cuda()
        self._tokenizer, self._image_processor, self._max_length = (
            llmc_model.tokenizer,
            llmc_model.image_processor,
            llmc_model.max_length,
        )

        del llmc_model.tokenizer
        del llmc_model.image_processor
        del llmc_model.max_length

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert (
            self.batch_size_per_gpu == 1
        ), 'Llava currently does not support batched generation.'

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
                    'Detected that you are using DistributedType.DEEPSPEED.'
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
