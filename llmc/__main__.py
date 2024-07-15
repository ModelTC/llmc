from loguru import logger
import argparse
import torch
import os
from llmc.data import BaseTokenizer, BaseDataset
from llmc.models import *
from llmc.compression.quantization import *
from llmc.compression.sparsification import *
from llmc.utils.registry_factory import ALGO_REGISTRY, MODEL_REGISTRY
from llmc.eval import PerplexityEval
import gc
import yaml
from easydict import EasyDict
from llmc.utils import seed_all, check_config, mkdirs
import copy
import time
import json


def main(config):
    tokenizer = BaseTokenizer(config.model.path, config.model.tokenizer_mode)
    model = MODEL_REGISTRY[config.model.type](
        config.model.path, config.model.torch_dtype
    )

    logger.info(tokenizer)
    logger.info(model)

    if "eval" in config and len(config.eval.eval_pos):
        eval_list = []
        name_list = (
            config.eval.name
            if not isinstance(config.eval.name, str)
            else [config.eval.name]
        )
        for name in name_list:
            eval_config = copy.deepcopy(config.eval)
            eval_config.name = name
            if len(name_list) != 1:  # eval multi datasets
                eval_config.path = os.path.join(config.eval.path, name)
            ppl_eval = PerplexityEval(tokenizer.get_tokenizer(), eval_config)
            eval_list.append(ppl_eval)

    if "eval" in config and "pretrain" in config.eval.eval_pos:
        for ppl_eval in eval_list:
            ppl = ppl_eval.eval(model)
            logger.info(f"{ppl_eval.dataset} ppl : {ppl}")
    sparsification = None
    if not config.get("calib", False):
        blockwise_opt = ALGO_REGISTRY[config.quant.method](
            model, quant_config=config.quant, config=config
        )
        blockwise_opt.run_block_loop()
    else:
        dataset = BaseDataset(tokenizer.get_tokenizer(), config.calib)
        calib_data = dataset.get_calib_dataset()
        model.collect_first_block_input(calib_data)
        del calib_data
        gc.collect()
        torch.cuda.empty_cache()
        if not config.get("sparse", False):
            sparsification = False
            blockwise_opt = ALGO_REGISTRY[config.quant.method](
                model, config.quant, model.get_first_block_input(), config
            )
        else:
            sparsification = True
            blockwise_opt = ALGO_REGISTRY[config.sparse.method](
                model, config.sparse, model.get_first_block_input(), config
            )
        blockwise_opt.run_block_loop()

        if "eval" in config and "transformed" in config.eval.eval_pos:
            if not sparsification:
                blockwise_opt.deploy("origin_float")
            for ppl_eval in eval_list:
                ppl = ppl_eval.eval(model)
                logger.info(f"{ppl_eval.dataset} ppl : {ppl}")

        if "cvt" in config and config.get("cvt", True):
            blockwise_opt.run_block_cvt()

        if "save" in config and config.save.get("save_trans", False):
            blockwise_opt.save_model(save_trans_path)

        if "save" in config and config.save.get("save_trtllm", False):
            blockwise_opt.save_model(save_trtllm_trans_path)
            from llmc.utils.export_trtllm import cvt_trtllm_engine

            cvt_trtllm_engine(
                save_trtllm_trans_path,
                save_trtllm_engine_path,
                config.save.get("trtllm_cfg"),
            )

    if "eval" in config and "fake_quant" in config.eval.eval_pos:
        blockwise_opt.deploy("fake_quant")
        for ppl_eval in eval_list:
            ppl = ppl_eval.eval(model)
            logger.info(f"{ppl_eval.dataset} ppl : {ppl}")

    if "save" in config and config.save.get("save_fake", False):
        blockwise_opt.deploy("fake_quant")
        blockwise_opt.save_model(save_fake_path)

    if "save" in config and config.save.get("save_lightllm", False):
        blockwise_opt.deploy("real_quant")
        blockwise_opt.save_model(save_quant_path)


if __name__ == "__main__":
    llmc_start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config = EasyDict(config)

    check_config(config)

    logger.info(f"args: {args}")
    logger.info(f"config:\n{json.dumps(config, ensure_ascii=False, indent=4)}")

    seed_all(config.base.seed)

    # mkdirs
    if "save" in config:
        if config.save.get("save_trans", False):
            save_trans_path = os.path.join(config.save.save_path, "transformed_model")
            mkdirs(save_trans_path)
        if config.save.get("save_trtllm", False):
            save_trtllm_trans_path = os.path.join(
                config.save.save_path, "trtllm_transformed_model"
            )
            mkdirs(save_trtllm_trans_path)
            save_trtllm_engine_path = os.path.join(
                config.save.save_path, "trtllm_engine"
            )
            mkdirs(save_trtllm_engine_path)
        if config.save.get("save_lightllm", False):
            save_quant_path = os.path.join(config.save.save_path, "real_quant_model")
            mkdirs(save_quant_path)
        if config.save.get("save_fake", False):
            save_fake_path = os.path.join(config.save.save_path, "fake_quant_model")
            mkdirs(save_fake_path)

    main(config)

    llmc_end_time = time.time()
    llmc_duration_time = llmc_end_time - llmc_start_time
    logger.info(f"llmc_duration_time: {llmc_duration_time} s")
    logger.info("--- llmc finished ---")
