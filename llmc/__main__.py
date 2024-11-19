import argparse
import copy
import gc
import json
import os
import time

import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from loguru import logger
from torch.distributed import destroy_process_group, init_process_group

from llmc.compression.quantization import *
from llmc.compression.sparsification import *
from llmc.data import BaseDataset
from llmc.eval import (AccuracyEval, PerplexityEval, TokenConsistencyEval,
                       VLMEval)
from llmc.models import *
from llmc.utils import (check_config, mkdirs, print_important_package_version,
                        seed_all, update_autoawq_quant_config,
                        update_vllm_quant_config)
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY


def main(config):
    model = MODEL_REGISTRY[config.model.type](config)

    logger.info(f'model: {model}')
    logger.info(f'tokenizer: {model.get_tokenizer()}')

    if int(os.environ['RANK']) == 0:
        if 'eval' in config and len(config.eval.eval_pos):
            eval_list = []
            name_list = (
                config.eval.name
                if not isinstance(config.eval.name, str)
                else [config.eval.name]
            )
            for name in name_list:
                config_for_eval = copy.deepcopy(config)
                config_for_eval.eval.name = name
                if len(name_list) != 1:  # eval multi datasets
                    config_for_eval.eval.path = os.path.join(config.eval.path, name)
                if config.eval.type == 'acc':
                    acc_eval = AccuracyEval(config_for_eval)
                    eval_list.append(acc_eval)
                elif config.eval.type == 'img_txt':
                    acc_eval = VLMEval(config_for_eval)
                    eval_list.append(acc_eval)
                else:
                    ppl_eval = PerplexityEval(model.get_tokenizer(), config_for_eval)
                    eval_list.append(ppl_eval)

        if 'eval' in config and 'pretrain' in config.eval.eval_pos:
            if config.eval.type == 'acc':
                for acc_eval in eval_list:
                    acc = acc_eval.eval(model)
                    logger.info(f'{config.eval.name} acc : {acc}')
            elif config.eval.type == 'img_txt':
                for vlm_eval in eval_list:
                    results = vlm_eval.eval(model)
                    logger.info(f'{config.eval.name} results : {results}')
            else:
                for ppl_eval in eval_list:
                    ppl = ppl_eval.eval(model)
                    logger.info(f'{ppl_eval.dataset} ppl : {ppl}')
    for modality in config.quant.get('quant_objects', ['language']):
        if not config.get('calib', False):
            blockwise_opt = ALGO_REGISTRY[config.quant.method](
                model,
                quant_config=config.quant,
                input=None,
                padding_mask=None,
                config=config,
                modality=modality,
            )
            blockwise_opt.run_block_loop()
            dist.barrier()
        else:
            dataset = BaseDataset(model.get_tokenizer(), config.calib, model.batch_process)
            calib_data, padding_mask = dataset.get_calib_dataset()
            padding_side = getattr(model.get_tokenizer(), 'padding_side', None)
            model.collect_first_block_input(calib_data,
                                            padding_mask,
                                            padding_side,
                                            config.calib.type,
                                            modality)
            del calib_data
            gc.collect()
            torch.cuda.empty_cache()
            if not config.get('sparse', False):
                blockwise_opt = ALGO_REGISTRY[config.quant.method](
                    model,
                    config.quant,
                    model.get_first_block_input(),
                    model.get_padding_mask(),
                    config,
                    modality
                )
            else:
                blockwise_opt = ALGO_REGISTRY[config.sparse.method](
                    model,
                    config.sparse,
                    model.get_first_block_input(),
                    model.get_padding_mask(),
                    config,
                    modality
                )
            blockwise_opt.run_block_loop()
            dist.barrier()

    if int(os.environ['RANK']) == 0:
        if 'eval' in config and 'transformed' in config.eval.eval_pos:
            blockwise_opt.deploy('origin_float')
            if config.eval.type == 'acc':
                for acc_eval in eval_list:
                    acc = acc_eval.eval(model)
                    logger.info(f'{config.eval.name} acc : {acc}')
            elif config.eval.type == 'img_txt':
                for vlm_eval in eval_list:
                    results = vlm_eval.eval(model)
                    logger.info(f'{config.eval.name} results : {results}')
            else:
                for ppl_eval in eval_list:
                    ppl = ppl_eval.eval(model)
                    logger.info(f'{ppl_eval.dataset} ppl : {ppl}')

        if 'save' in config and config.save.get('save_trans', False):
            blockwise_opt.save_model(save_trans_path)

        if 'save' in config and config.save.get('save_trtllm', False):
            blockwise_opt.save_model(save_trtllm_trans_path)
            from llmc.utils.export_trtllm import cvt_trtllm_engine

            cvt_trtllm_engine(
                save_trtllm_trans_path,
                save_trtllm_engine_path,
                config.save.get('trtllm_cfg'),
            )

        if 'eval' in config and 'fake_quant' in config.eval.eval_pos:
            blockwise_opt.deploy('fake_quant')
            if config.eval.type == 'acc':
                for acc_eval in eval_list:
                    acc = acc_eval.eval(model)
                    logger.info(f'{config.eval.name} acc : {acc}')
            elif config.eval.type == 'img_txt':
                for vlm_eval in eval_list:
                    results = vlm_eval.eval(model)
                    logger.info(f'{config.eval.name} results : {results}')
            else:
                for ppl_eval in eval_list:
                    ppl = ppl_eval.eval(model)
                    logger.info(f'{ppl_eval.dataset} ppl : {ppl}')

            if 'eval_token_consist' in config.eval and config.eval.eval_token_consist:
                org_model = MODEL_REGISTRY[config.model.type](config)
                token_consist_eval = TokenConsistencyEval(model.get_tokenizer(),
                                                          config_for_eval)
                consistency_ratio = token_consist_eval.eval(model, org_model)
                logger.info(f'Token consistency ratio: {consistency_ratio}')
                del org_model

        if 'save' in config and config.save.get('save_fake', False):
            blockwise_opt.deploy('fake_quant')
            blockwise_opt.save_model(save_fake_path)

        if 'save' in config and config.save.get('save_vllm', False):
            w, a = config.quant.weight, config.quant.get('act')
            if isinstance(w.bit, str):
                assert a, 'Only WA float quant is supported.'
                assert w.symmetric and a.symmetric, 'Only symmetric quant is supported.'
                assert w.bit == a.bit and w.bit in ['e4m3', 'e5m2'] and \
                    a.bit in ['e4m3', 'e5m2'], 'Only WA FP8 quant is supported'
            else:
                assert w.symmetric, 'Only symmetric quant is supported.'
                assert w.bit in [4, 8], 'Supported quant: w4a16, w8a16, w8a8.'
                if a:
                    assert a.symmetric, 'Only symmetric quant is supported.'
                    assert a.bit == 8, 'Supported quant: w4a16, w8a16, w8a8.'
            blockwise_opt.deploy('vllm_quant')
            blockwise_opt.save_model(save_quant_path)
            update_vllm_quant_config(blockwise_opt.model, config, save_quant_path)

        if 'save' in config and config.save.get('save_sgl', False):
            w, a = config.quant.weight, config.quant.get('act')
            if isinstance(w.bit, str):
                assert a, 'Only WA float quant is supported.'
                assert w.symmetric and a.symmetric, 'Only symmetric quant is supported.'
                assert w.bit == a.bit and w.bit in ['e4m3', 'e5m2'] and \
                    a.bit in ['e4m3', 'e5m2'], 'Only WA FP8 quant is supported'
            else:
                assert w.symmetric, 'Only symmetric quant is supported.'
                assert w.bit in [4, 8], 'Supported quant: w4a16, w8a16, w8a8.'
                if a:
                    assert a.symmetric, 'Only symmetric quant is supported.'
                    assert a.bit == 8, 'Supported quant: w4a16, w8a16, w8a8.'
            blockwise_opt.deploy('sgl_quant')
            blockwise_opt.save_model(save_quant_path)
            update_vllm_quant_config(blockwise_opt.model, config, save_quant_path)

        if 'save' in config and config.save.get('save_autoawq', False):
            assert config.quant.weight.bit in [4] and 'act' not in config.quant, \
                'AutoAWQ supports only 4-bit weight-only quantization.'
            assert not config.quant.weight.symmetric, 'Only asymmetric quant is supported.'

            blockwise_opt.deploy('autoawq_quant')
            blockwise_opt.save_model(save_quant_path)
            update_autoawq_quant_config(config, save_quant_path)

        if 'save' in config and config.save.get('save_mlcllm', False):
            assert config.quant.weight.bit in [4] and 'act' not in config.quant, \
                'MlcLLM supports only 4-bit weight-only quantization.'
            assert not config.quant.weight.symmetric, 'Only asymmetric quant is supported.'

            blockwise_opt.deploy('mlcllm_quant')
            blockwise_opt.save_model(save_quant_path)
            update_autoawq_quant_config(config, save_quant_path)

        if 'opencompass' in config:
            assert config.save.get('save_trans', False)
            cfg_path = config['opencompass']['cfg_path']
            output_path = config['opencompass']['output_path']
            eval_model_path = os.path.abspath(save_trans_path)
            opencompass_cmd = (
                f'opencompass {cfg_path} -w {output_path} '
                f'--llmc_cfg {args.config} '
                f'--llmc_eval_mode quant '
                f'--llmc_model_path {eval_model_path}'
            )
            logger.info(f'opencompass_cmd : {opencompass_cmd}')
            os.system(opencompass_cmd)
    dist.barrier()


if __name__ == '__main__':
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task_id', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    if int(os.environ['RANK']) != 0:
        logger.remove()

    check_config(config)

    logger.info(f'args: {args}')
    logger.info(f'config:\n{json.dumps(config, ensure_ascii=False, indent=4)}')

    print_important_package_version()

    logger.info(f'WORLD_SIZE : {int(os.environ["WORLD_SIZE"])}')

    seed_all(config.base.seed + int(os.environ['RANK']))

    # Ensure only the main process creates directories
    if int(os.environ['RANK']) == 0:
        if 'save' in config:
            if config.save.get('save_trans', False):
                save_trans_path = os.path.join(config.save.save_path, 'transformed_model')
                mkdirs(save_trans_path)
            if config.save.get('save_trtllm', False):
                save_trtllm_trans_path = os.path.join(
                    config.save.save_path, 'trtllm_transformed_model'
                )
                mkdirs(save_trtllm_trans_path)
                save_trtllm_engine_path = os.path.join(
                    config.save.save_path, 'trtllm_engine'
                )
                mkdirs(save_trtllm_engine_path)
            if config.save.get('save_vllm', False):
                save_quant_path = os.path.join(config.save.save_path, 'vllm_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_sgl', False):
                save_quant_path = os.path.join(config.save.save_path, 'sgl_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_autoawq', False):
                save_quant_path = os.path.join(config.save.save_path, 'autoawq_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_mlcllm', False):
                save_quant_path = os.path.join(config.save.save_path, 'mlcllm_quant_model')
                mkdirs(save_quant_path)
            if config.save.get('save_fake', False):
                save_fake_path = os.path.join(config.save.save_path, 'fake_quant_model')
                mkdirs(save_fake_path)

    # Synchronize all processes after directory creation
    dist.barrier()

    main(config)

    destroy_process_group()

    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f'llmc_duration_time: {llmc_duration_time} s')
    logger.info('--- llmc finished ---')
