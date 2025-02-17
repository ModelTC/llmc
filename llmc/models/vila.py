import types
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from loguru import logger
from tqdm import tqdm

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel

try:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (KeywordsStoppingCriteria,
                                get_model_name_from_path, process_images,
                                tokenizer_image_token)
    from llava.model.builder import load_pretrained_model
except ImportError as e:
    logger.warning(
        f'VILA is not installed. Please install VILA to use this model. Error: {e}'
    )


@MODEL_REGISTRY
class Vila(BaseModel):

    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_tokenizer(self):
        # init in build_model, do nothing
        pass

    def build_model(self):
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.mm_model, self.image_processor, self.max_length = \
            load_pretrained_model(self.model_path, model_name, attn_implementation='eager')
        self.mm_model_config = self.mm_model.config
        if not self.use_cache:
            self.mm_model_config.llm_cfg['use_cache'] = False
        if self.mm_model.dtype == torch.float16:
            self.mm_model_config.llm_cfg['torch_dtype'] = 'float16'
            self.mm_model_config.vision_tower_cfg['torch_dtype'] = 'float16'
        logger.info(f'self.mm_model_config : {self.mm_model_config}')
        self.mm_model.image_processor = self.image_processor
        self.vlm_model = self.mm_model
        self.eval_name = 'VilaEval'
        logger.info(f'self.mm_model : {self.mm_model}')
        self.vision_model = self.mm_model.get_vision_tower()  # vit model
        self.mm_projector = self.mm_model.get_mm_projector()
        self.model = self.mm_model.get_llm()  # llm model
        self.model_config = types.SimpleNamespace(
            **self.mm_model_config.llm_cfg)
        self.conv_template = conv_templates['vicuna_v1']

    def batch_process(self,
                      img_qas,
                      calib_or_eval='eval',
                      apply_chat_template=True,
                      return_inputs=True):  # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        add_answer = calib_or_eval == 'calib' and self.config['calib'].get(
            'add_answer', False)
        batch_size = len(img_qas)
        assert batch_size == 1, f'Batch size should be 1 for Vila, but got {batch_size}.'

        question = img_qas[0]['question']
        answer = img_qas[0]['answer']
        message = '<image>\n ' + question
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], answer if add_answer else None)
        text = conv.get_prompt()

        if calib_or_eval == 'calib':
            logger.info(f'Calib data is:\n{text}')
        if not return_inputs:
            return text
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors='pt',
        )
        input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id \
            is not None else self.tokenizer.eos_token_id
        attention_masks = input_ids.ne(pad_token_ids).long().cuda()

        img_path = img_qas[0]['image']
        images = process_images([img_path], self.image_processor,
                                self.mm_model.config).to(self.mm_model.device,
                                                         dtype=torch.float16)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer,
                                                     input_ids)
        inputs = {
            'input_ids': input_ids,
            'images': images,
            'attention_mask': attention_masks,
            'stopping_criteria': [stopping_criteria]
        }
        return inputs

    def find_blocks(self):
        assert self.get_modality() == 'language'
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens
        self.rotary_emb = self.model.model.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'
        self.pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_attention_rotary_layers(self):
        return [self.rotary_emb]

    def get_attn_in_block(self, block):
        return {'self_attn': block.self_attn}

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.mm_projector.layers[-1]]

    def get_layers_except_blocks(self):
        return [
            self.embed_tokens, self.rotary_emb, self.model.model.norm,
            self.model.lm_head
        ]  # noqa

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        assert self.get_modality() == 'language'
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        assert self.get_modality() == 'language'
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
                'layers': {
                    'self_attn.o_proj': block.self_attn.o_proj
                },
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
                'layers': {
                    'mlp.down_proj': block.mlp.down_proj
                },
                'prev_op': [block.mlp.up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]


@MODEL_REGISTRY
class VilaEval(lmms):

    def __init__(
        self,
        llmc_model,
        pretrained: str = 'Efficient-Large-Model/VILA1.5-40b',
        max_frames_num: Optional[int] = 100,
        truncation: Optional[bool] = True,
        device: Optional[str] = 'cuda:0',
        batch_size: Optional[Union[int, str]] = 1,
        device_map='cuda:0',
        use_cache=True,
        # whether to truncate the context in generation, set it False for LLaVA-1.6
        truncate_context=False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f'Unexpected kwargs: {kwargs}'
        assert batch_size == 1, f'Batch size should be 1 for Vila, but got {batch_size}.'

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(
            weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(
                f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'
        elif accelerator.num_processes == 1 and device_map == 'auto':
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(
                f'cuda:{accelerator.local_process_index}')
            self.device_map = f'cuda:{accelerator.local_process_index}'

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = max_frames_num
        # self._config = AutoConfig.from_pretrained(self.pretrained)
        self._tokenizer = llmc_model.tokenizer
        self._model = llmc_model
        self.image_processor = llmc_model.image_processor
        self._max_length = 2048
        self._config = llmc_model.config

        self._model.eval().cuda()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_templates['vicuna_v1']
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation.
        # See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED
            ], 'Unsupported distributed type provided. Only DDP and FSDP are supported.'
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config
            # before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the
            # prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works,
            # but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    'train_micro_batch_size_per_gpu':
                    self.batch_size_per_gpu,
                    'train_batch_size':
                    self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs)
                logger.info(
                    'Detected that you are using DistributedType.DEEPSPEED. Make sure you run '
                    '`accelerate config` and set zero stage to 0'
                )
            if accelerator.distributed_type == DistributedType.FSDP or \
                    accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model,
                                                        evaluation_mode=True)
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
            self._word_size = 1
        else:
            logger.info(f'Using single device: {self._device}')
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, 'accelerator'):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing
        # than end of *sentence*
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self,
                      requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError('Loglikelihood is not implemented for Vila')

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests),
                    disable=(self.rank != 0),
                    desc='Model Responding')

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
                reg.args for reg in requests
        ]:
            # encode, pad, and truncate contexts for this batch
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            assert len(visuals) == 1
            image = process_images(visuals, self.image_processor,
                                   self._config).to(self.device,
                                                    dtype=torch.float16)

            qs = '<image>\n ' + contexts

            conv = self.conv_template.copy()

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                self._tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt').unsqueeze(0).cuda()
            pad_token_ids = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id \
                is not None else self._tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).long().cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self._tokenizer, input_ids)

            if 'max_new_tokens' not in gen_kwargs:
                gen_kwargs['max_new_tokens'] = 1024
            if 'temperature' not in gen_kwargs:
                gen_kwargs['temperature'] = 0.2
            if 'top_p' not in gen_kwargs:
                gen_kwargs['top_p'] = None
            if 'num_beams' not in gen_kwargs:
                gen_kwargs['num_beams'] = 1

            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    images=image,
                    attention_mask=attention_masks,
                    use_cache=self.use_cache,
                    stopping_criteria=[stopping_criteria],
                    do_sample=True if gen_kwargs['temperature'] > 0 else False,
                    temperature=gen_kwargs['temperature'],
                    top_p=gen_kwargs['top_p'],
                    num_beams=gen_kwargs['num_beams'],
                    max_new_tokens=gen_kwargs['max_new_tokens'],
                )

            outputs = self._tokenizer.batch_decode(
                output_ids, skip_special_tokens=True)[0].strip()
            # print("Question: ", contexts)
            # print("Answer: ", outputs)
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError('TODO: Implement multi-round generation')
