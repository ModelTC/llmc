from abc import abstractmethod, ABCMeta
import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from loguru import logger
from functools import partial
from collections import defaultdict
from llmc.compression.quantization.utils import check_do_quant
from llmc.compression.quantization.module_utils import (
    _LLMC_LN_TYPES_,
    _TRANSFORMERS_LN_TYPES_,
)
from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_,
    _TRANSFORMERS_LINEAR_TYPES_,
)
from llmc.compression.quantization.utils import (
    get_wquantizer,
    get_aquantizer,
    check_w_only,
)


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_path, torch_dtype):
        self.model_path = model_path
        self.torch_dtype = torch_dtype if torch_dtype == "auto" else eval(torch_dtype)
        self.build_model()
        self.model.eval()
        self.find_blocks()
        self.find_embed_layers()
        self.find_block_name()

    @abstractmethod
    def find_blocks(self):
        pass

    def find_block_name(self):
        pass

    def get_model(self):
        return self.model

    def get_blocks(self):
        return self.blocks

    @abstractmethod
    def find_embed_layers(self):
        pass

    @abstractmethod
    def get_embed_layers(self):
        pass

    @abstractmethod
    def get_layers_except_blocks(self):
        pass

    @abstractmethod
    def get_subsets_in_block(self, block):
        pass

    @abstractmethod
    def has_bias(self):
        pass

    def __str__(self):
        return f"\nConfig: \n{str(self.model_config)} \nModel: \n{str(self.model)}"

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if hasattr(self.model_config, "use_cache"):
            self.model_config.use_cache = False
        logger.info(f"self.model_config : {self.model_config}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )

    @torch.no_grad()
    def collect_first_block_input(self, calib_data):
        first_block_input = defaultdict(list)

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                first_block_input["data"].append(inp)
                if "output_router_logits" in kwargs:
                    assert kwargs["output_router_logits"] == False
                    kwargs.pop("output_router_logits")
                first_block_input["kwargs"].append(kwargs)
                raise ValueError

        self.move_embed_to_device("cuda")
        self.blocks[0] = self.blocks[0].cuda()
        self.blocks[0] = Catcher(self.blocks[0])

        for data in calib_data:
            try:
                self.model(data.to(next(self.model.parameters()).device))
            except ValueError:
                pass
        self.first_block_input = first_block_input
        self.blocks[0] = self.blocks[0].module
        self.blocks[0] = self.blocks[0].cpu()
        self.move_embed_to_device("cpu")

    def get_first_block_input(self):
        return self.first_block_input

    def get_model_config(self):
        return self.model_config

    def move_embed_to_device(self, device):
        for embed_layer in self.get_embed_layers():
            embed_layer = embed_layer.to(device)

    def get_block_linears(self, block):
        return {
            name: m
            for name, m in block.named_modules()
            if isinstance(m, tuple(_LLMC_LINEAR_TYPES_ + _TRANSFORMERS_LINEAR_TYPES_))
        }

    def replace_module_all(self, module, params_dict):
        for block_idx in range(len(self.blocks)):
            logger.info(f"Replace block index: {block_idx}/{len(self.blocks)}")
            block = self.blocks[block_idx]
            block = block.cuda()
            self.replace_module_block(module, block, block_idx, params_dict)
            block = block.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"The Replaced model: {self.model}")

    def replace_module_block(self, module, block, block_idx, params_dict):
        if module in _LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_:
            layer_norms = self.get_layernorms_in_block(block)
            self.replace_module_layernorm(
                module, block, layer_norms, block_idx, params_dict
            )
        else:
            subset = {}
            subset["layers"] = self.get_block_linears(block)
            self.replace_module_subset(module, block, subset, block_idx, params_dict)

    def replace_module_subset(self, module, block, subset, block_idx, params_dict):
        layers_dict = subset["layers"]

        for name, m in layers_dict.items():
            if isinstance(m, module):
                logger.info(f"{name} in {block_idx}-th block. Target module is same. No need to replace this layer.")
                continue
            if not params_dict.get("mix_bits", False):
                logger.info(f"replace >>> {name} in {block_idx}-th block")
                params_dict_tmp = {}
                params_dict_tmp["a_qdq"] = params_dict["a_qdq"]
                params_dict_tmp["w_qdq"] = params_dict["w_qdq"]
                M = module.new(m, **params_dict_tmp)
            else:
                # mix bits
                if not check_do_quant(
                    block_idx,
                    name,
                    params_dict["mix_bits_map"],
                    params_dict["quantizer_mix_bits"],
                ):
                    logger.info(
                        f"This layer {name} in {block_idx}-th block is set to float. No need to replace this layer."
                    )
                    continue
                else:
                    logger.info(f"replace >>> {name} in {block_idx}-th block")
                params_dict_tmp = {}
                params_dict_tmp["debug_print"] = {}
                wquantizer = get_wquantizer(
                    block_idx,
                    name,
                    params_dict["mix_bits_map"],
                    params_dict["quantizer_mix_bits"],
                    params_dict["wquantizer_default"],
                )
                params_dict_tmp["w_qdq"] = partial(
                    params_dict["w_qdq"], wquantizer=wquantizer
                )
                params_dict_tmp["debug_print"]["weight"] = {}
                params_dict_tmp["debug_print"]["weight"]["bit"] = wquantizer.bit
                params_dict_tmp["debug_print"]["weight"]["sym"] = wquantizer.sym
                params_dict_tmp["debug_print"]["weight"][
                    "granularity"
                ] = wquantizer.granularity
                if wquantizer.granularity == "per_group":
                    params_dict_tmp["debug_print"]["weight"][
                        "group_size"
                    ] = wquantizer.group_size
                if not check_w_only(
                    block_idx,
                    name,
                    params_dict["mix_bits_map"],
                    params_dict["quantizer_mix_bits"],
                    params_dict["w_only_default"],
                ):
                    aquantizer = get_aquantizer(
                        block_idx,
                        name,
                        params_dict["mix_bits_map"],
                        params_dict["quantizer_mix_bits"],
                        params_dict["aquantizer_default"],
                    )
                    params_dict_tmp["a_qdq"] = partial(
                        params_dict["a_qdq"], aquantizer=aquantizer
                    )
                    params_dict_tmp["debug_print"]["act"] = {}
                    params_dict_tmp["debug_print"]["act"]["bit"] = aquantizer.bit
                    params_dict_tmp["debug_print"]["act"]["sym"] = aquantizer.sym
                    params_dict_tmp["debug_print"]["act"][
                        "granularity"
                    ] = aquantizer.granularity
                else:
                    params_dict_tmp["a_qdq"] = None
                M = module.new(m, **params_dict_tmp)

            name_tmp = name.rsplit(".", 1)
            if len(name_tmp) == 2:
                parent_name = name_tmp[0]
                parent = block.get_submodule(parent_name)
                child_name = name_tmp[1]
            elif len(name_tmp) == 1:
                parent = block
                child_name = name_tmp[0]

            setattr(parent, child_name, M)

    def replace_module_layernorm(self, module, block, lns, i, params_dict):
        for name, m in lns.items():
            if isinstance(m, module):
                continue
            M = module.new(m, **params_dict)

            name_tmp = name.rsplit(".", 1)
            if len(name_tmp) == 2:
                parent_name = name_tmp[0]
                parent = block.get_submodule(parent_name)
                child_name = name_tmp[1]
            elif len(name_tmp) == 1:
                parent = block
                child_name = name_tmp[0]

            setattr(parent, child_name, M)

    def convert_dtype(self, dtype="torch.float16"):
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].to(dtype)
