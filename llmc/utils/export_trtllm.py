import os

from loguru import logger
from tensorrt_llm._utils import release_gc
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import LLaMAForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def preload_model(model_dir, load_model_on_cpu):
    use_safetensors = True
    from transformers import AutoConfig, AutoModelForCausalLM

    # hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    use_safetensors = (
        any([f.endswith('.safetensors') for f in os.listdir(model_dir)])
        and use_safetensors
    )
    if use_safetensors:
        return None
    model = model_cls.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        torch_dtype='auto',
        trust_remote_code=True,
    )
    return model


def args_to_build_options():
    return {
        'use_parallel_embedding': False,
        'embedding_sharding_dim': 0,
        'share_embedding_table': False,
        'disable_weight_only_quant_plugin': False,
    }


def convert_and_save_hf(hf_model, output_dir, cfg):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_dir = hf_model
    load_model_on_cpu = False
    load_by_shard = False
    dtype = 'float16'
    # workers = 1
    world_size = cfg['tp_size'] * cfg['pp_size']
    # Need to convert the cli args to the kay-value pairs and
    # override them in the generate config dict.
    # Ideally these fields will be moved out of the config and pass them into build API,
    # keep them here for compatibility purpose for now,
    # before the refactor is done.
    override_fields = {'moe_tp_mode': MoeConfig.ParallelismMode.TENSOR_PARALLEL}

    quant_config = QuantConfig()
    quant_config.exclude_modules = ['lm_head']
    quant_config.quant_algo = QuantAlgo.W4A16
    quantization = quant_config
    override_fields.update(args_to_build_options())

    hf_model = (
        preload_model(model_dir, load_model_on_cpu) if not load_by_shard else None
    )

    def convert_and_save_rank(cfg, rank):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=cfg['tp_size'],
            pp_size=cfg['pp_size'],
        )
        llama = LLaMAForCausalLM.from_hugging_face(
            model_dir,
            dtype,
            mapping=mapping,
            quantization=quantization,
            load_by_shard=load_by_shard,
            load_model_on_cpu=load_model_on_cpu,
            override_fields=override_fields,
            preloaded_model=hf_model,
        )
        llama.save_checkpoint(output_dir, save_config=(rank == 0))
        del llama

    convert_and_save_rank(cfg, rank=0)
    release_gc()


def cvt_trtllm_engine(hf_model, export_engine_dir, cfg):
    logger.info('Start to export trtllm engine...')
    output_dir = hf_model + '_trtllm_tmp'
    convert_and_save_hf(hf_model, output_dir, cfg)
    cmd = (
        f'trtllm-build '
        f'--checkpoint_dir {output_dir} '
        f'--output_dir {export_engine_dir} '
        f'--gemm_plugin float16'
    )
    os.system(cmd)
    cmd = f'rm -r {output_dir}'
    os.system(cmd)
    logger.info(f'trtllm engine path: {export_engine_dir}')


if __name__ == '__main__':
    cfg = {
        'tp_size': 1,
        'pp_size': 1,
    }
    hf_model = ''
    export_engine_dir = './output_engine'
    cvt_trtllm_engine(hf_model, export_engine_dir, cfg)
