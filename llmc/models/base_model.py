from abc import abstractmethod, ABCMeta
import gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from loguru import logger
from collections import defaultdict
from llmc.compression.quantization.module_utils import (
    FakeQuantLinear,
    EffcientFakeQuantLinear,
    RealQuantLinear,
    OriginFloatLinear,
    LlmcLayerNorm,
    LlmcLlamaRMSNorm,
    LlmcMistralRMSNorm,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralRMSNorm


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
            if isinstance(
                m,
                (
                    nn.Linear,
                    FakeQuantLinear,
                    EffcientFakeQuantLinear,
                    RealQuantLinear,
                    OriginFloatLinear,
                ),
            )
        }

    def replace_module_all(self, module, params_dict):
        for i in range(len(self.blocks)):
            logger.info(f"Replace block index: {i+1}/{len(self.blocks)}")
            block = self.blocks[i]
            block = block.cuda()
            self.replace_module_block(module, block, i, params_dict)
            block = block.cpu()

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"The Replaced model: {self.model}")

    def replace_module_block(self, module, block, i, params_dict):
        if module in [LlmcLayerNorm, LlmcLlamaRMSNorm, LlmcMistralRMSNorm]:
            layer_norms = self.get_layernorms_in_block(block)
            self.replace_module_layernorm(module, block, layer_norms, i, params_dict)
        else:
            subset = {}
            subset["layers"] = self.get_block_linears(block)
            self.replace_module_subset(module, block, subset, i, params_dict)

    def replace_module_subset(self, module, block, subset, i, params_dict):
        layers_dict = subset["layers"]

        for name, m in layers_dict.items():
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
