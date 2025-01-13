import os
import random
import shutil

import numpy as np
import torch
from loguru import logger


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_config(config):
    if config.get('sparse', False):
        logger.info('Use sparsificatino method')
    else:

        def check_weight_setting(weight_setting):
            if weight_setting.granularity == 'per_group':
                assert weight_setting.group_size > 0
            elif weight_setting.granularity == 'per_head':
                assert weight_setting.head_num > 0

        for _, modality_config in config.quant.items():
            if not isinstance(modality_config, dict) or not modality_config.get('weight', False):
                continue
            if modality_config.weight.get('granularity', False):
                weight_setting = modality_config.weight
                check_weight_setting(weight_setting)
            if modality_config.weight.get('w_1', False):
                weight_setting = modality_config.weight.w_1
                check_weight_setting(weight_setting)
            if modality_config.weight.get('w_2', False):
                weight_setting = modality_config.weight.w_2
                check_weight_setting(weight_setting)
    if config.model.get('tokenizer_mode', False):
        assert (
            config.model.tokenizer_mode == 'slow'
            or config.model.tokenizer_mode == 'fast'
        ), 'Tokenizer_mode should be slow or fast.'
        logger.info(f'Tokenizer_mode is set to {config.model.tokenizer_mode}.')
    else:
        config.model.tokenizer_mode = 'slow'
        logger.info('Tokenizer_mode is set to slow.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception(f'{path} existed before. Need check.')


def copy_files(source_dir, target_dir, substring):
    for filename in os.listdir(source_dir):
        if substring in filename:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.copy(source_file, target_file)
            logger.info(f'Copied {filename} to {target_dir}')


def print_important_package_version():
    from importlib.metadata import version
    logger.info(f"torch : {version('torch')}")
    logger.info(f"transformers : {version('transformers')}")
    logger.info(f"tokenizers : {version('tokenizers')}")
    logger.info(f"huggingface-hub : {version('huggingface-hub')}")
    logger.info(f"datasets : {version('datasets')}")


def get_modality(config):
    modalities = []
    modality_configs = []
    compression_config = config.quant if 'quant' in config else config.sparse
    for modality in ['vision', 'language']:
        if modality in compression_config:
            compression_config[modality].modality = modality
            modalities.append(modality)
            modality_configs.append(compression_config[modality])
    if not modalities:
        compression_config.modality = 'language'
        return ['language'], [compression_config]
    return modalities, modality_configs


def deploy_all_modality(blockwise_opts, quant_format):
    for blockwise_opt in blockwise_opts:
        blockwise_opt.deploy(quant_format)
