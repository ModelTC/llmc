import argparse
import copy
import functools
import gc
import os
import sys

import torch
import yaml
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '../lm-evaluation-harness'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import lm_eval.__main__ as lm_eval
import torch.nn as nn
from easydict import EasyDict

from llmc.compression.quantization import FakeQuantLinear, Quantizer
from llmc.compression.quantization.base_blockwise_quantization import \
    BaseBlockwiseQuantization
from llmc.compression.quantization.module_utils import LlmcRMSNorm
from llmc.data import BaseDataset, BaseTokenizer
from llmc.eval import PerplexityEval
from llmc.models import *
from llmc.utils import check_config, mkdirs, seed_all
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY

if __name__ == '__main__':
    logger.warning('This script only supports transformed/original model type!')
    parser = lm_eval.setup_parser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--quarot', action='store_true')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)
    args.config = config
    if 'pretrained' not in args.model_args:
        if 'paralleize=True' in args.model_args:
            logger.error("Please remove 'paralleize=True' from model_args!")
            sys.exit(1)
        args.model_args += ',pretrained=' + config.model.path
    args.use_fast_tokenizer = config.model.tokenizer_mode == 'fast'
    lm_eval.cli_evaluate(args)
