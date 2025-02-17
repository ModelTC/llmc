from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from loguru import logger
from PIL import Image
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .minicpm import MiniCPM


@MODEL_REGISTRY
class MiniCPMV(MiniCPM):

    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.eval_name = 'MiniCPMVEval'
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True)
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.vlm_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype='auto',
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.vlm_model
        self.vlm_model_config = self.vlm_model.config
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')
        self.vision_model = self.vlm_model.vpm
        self.model = self.vlm_model.llm
        self.model_config = self.vlm_model_config
        self.processor = AutoProcessor.from_pretrained(self.model_path,
                                                       trust_remote_code=True)
        self.max_slice_nums = self.processor.image_processor.max_slice_nums
        self.max_length = 4096

    def batch_process(self,
                      img_qas,
                      calib_or_eval='eval',
                      apply_chat_template=True,
                      return_inputs=True):  # noqa
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert apply_chat_template
        add_answer = calib_or_eval == 'calib' and self.config['calib'].get(
            'add_answer', False)
        image_lists = []
        prompt_lists = []
        for idx in range(len(img_qas)):
            img_path = img_qas[idx]['image']
            question = img_qas[idx]['question']
            answer = img_qas[idx]['answer']
            image_lists.append([Image.open(img_path).convert('RGB')])
            if not add_answer:
                msg = [{
                    'role': 'user',
                    'content': '(<image>./</image>)\n' + question
                }]
            else:
                msg = [{
                    'role': 'user',
                    'content': '(<image>./</image>)\n' + question
                }, {
                    'role': 'assistant',
                    'content': answer
                }]
            prompt = self.processor.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            prompt_lists.append(prompt)
        if not return_inputs:
            return prompt_lists
        inputs = self.processor(
            prompt_lists,
            image_lists,
            max_slice_num=self.max_slice_nums,
            use_image_id=self.model_config.use_image_id,
            return_tensors='pt',
            max_length=self.max_length).to(self.vlm_model.device).to(
                next(self.vlm_model.parameters()).dtype)
        inputs.pop('image_sizes')
        inputs['tokenizer'] = self.processor.tokenizer
        return inputs

    def find_blocks(self):
        assert self.get_modality() == 'language'
        super().find_blocks()

    def get_layernorms_in_block(self, block):
        assert self.get_modality() == 'language'
        return super().get_layernorms_in_block(block)


@MODEL_REGISTRY
class MiniCPMVEval(lmms):
    """MiniCPM_V Model."""

    def __init__(
        self,
        llmc_model,
        pretrained: str = 'openbmb/MiniCPM-V',
        device: Optional[str] = 'cuda',
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        **kwargs,
    ) -> None:
        lmms.__init__(self)
        assert batch_size == 1, f'Batch size should be 1 for MiniCPMV, but got {batch_size}.'

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(
                f'cuda:{accelerator.local_process_index}')
        else:
            self._device = device
        self._model = llmc_model.eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        self._max_length = 4096
        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP, DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED
            ], 'Unsupported distributed type provided. Only DDP and FSDP are supported.'
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate
            # config before using the model
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
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model,
                                                        evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                logger.info(
                    f'Using {accelerator.num_processes} devices with data parallelism'
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
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
        return self.tokenizer.eos_token_id

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

    def tok_encode(self,
                   string: str,
                   left_truncate_len=None,
                   add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string,
                                         add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self,
                      requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, 'We have not implemented this function for MiniCPM_V yet'

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
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
            msgs = [{'role': 'user', 'content': [visuals[0], contexts]}]
            outputs = self.model.chat(image=None,
                                      msgs=msgs,
                                      tokenizer=self.tokenizer)
            res.append(outputs)
            pbar.update(1)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError('TODO: Implement multi-round generation')
