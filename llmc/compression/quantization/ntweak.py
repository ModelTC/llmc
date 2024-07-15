import torch
import torch.nn as nn
import functools
import gc
import pdb
import math

from math import inf
from loguru import logger
from tqdm import tqdm
from contextlib import nullcontext

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import FakeQuantLinear
from .module_utils import _LLMC_LN_TYPES_, _MODEL_LN_TYPES_PAIRS_
from .train_utils import NativeScalerWithGradNormCount, LossFunction
from llmc.utils.registry_factory import ALGO_REGISTRY


@ALGO_REGISTRY
class NormTweaking(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, config):
        super().__init__(model, quant_config, input, config)
        self.add_quant_config()

        self.attention_mask = self.input["kwargs"][0].get("attention_mask")
        self.position_ids = (
            self.input["kwargs"][0].get("position_ids")
            if model_type in ["Llama", "Mistral", "Qwen2"]
            else None
        )
        self.dev = torch.device("cuda")
        self.model_dtype = next(self.model.model.parameters()).dtype

    def add_quant_config(self):
        self.prefix = self.model.block_name_prefix
        self.loss_func = LossFunction(method="mse")
        self.deactive_amp = self.quant_config["special"]["deactive_amp"]

        if self.deactive_amp:
            self.dtype = torch.float
            self.traincast = nullcontext
        else:
            self.dtype = self.model_dtype
            self.traincast = torch.cuda.amp.autocast
        self.epochs = self.quant_config["special"]["epochs"]
        self.ntweak_lr = self.quant_config["special"]["ntweak_lr"]
        self.gamma = self.quant_config["special"]["gamma"]

    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input["data"]

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if "attention_mask" in self.input["kwargs"][i]:
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input["kwargs"][i])[0]
                    output.append(out)
        return output

    def get_original_out(self, block):
        if self.block_idx == 0:
            self.ori_out = self.block_forward(block)
        else:
            self.ori_out = self.block_forward(block, self.ori_out)

    def block_transform(self, block, input_feat, block_kwargs):
        logger.info(f"Start transform the {self.block_idx}-th block")

        with torch.no_grad():
            block.float()

        for i in range(len(self.input["data"])):
            self.input["data"][i] = self.input["data"][i].to(self.dtype)

        self.get_original_out(block)
        self.register_tweak_parameters(block)
        self.ntweak_train(block)

        self.apply_layer_norms(block)

        logger.info(f"End transform the {self.block_idx}-th block")

    def ntweak_train(self, block):
        optimizer = torch.optim.Adam([{"params": self.get_tweak_parameters(block)}])
        self.adjust_learning_rate(optimizer)

        for param_group in optimizer.param_groups:
            logger.info(param_group["lr"])

        loss_scaler = NativeScalerWithGradNormCount()

        for i in range(len(self.input["data"])):
            if self.deactive_amp:
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].repeat(self.input["data"][i].shape[0], 1, 1, 1)
            else:
                self.input["kwargs"][i]["attention_mask"] = (
                    self.input["kwargs"][i]["attention_mask"]
                    .repeat(self.input["data"][i].shape[0], 1, 1, 1)
                    .float()
                )

            self.input["data"][i] = self.input["data"][i].to(self.dtype)

        for epochs in range(self.epochs):
            loss_list = []
            norm_list = []

            for i in range(len(self.input["data"])):
                with self.traincast():
                    quant_out = block(self.input["data"][i], **self.input["kwargs"][i])[
                        0
                    ]

                    loss = self.loss_func(self.ori_out[i].to(self.dtype), quant_out)

                if not math.isfinite(loss.item()):
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()

                loss_list.append(loss.data)
                optimizer.zero_grad()
                norm = loss_scaler(
                    loss, optimizer, parameters=self.get_tweak_parameters(block)
                )
                norm_list.append(norm.data)

            loss_mean = torch.stack(loss_list).mean()
            norm_mean = torch.stack(norm_list).mean()
            logger.info(
                f"block {self.block_idx} iter {epochs} loss:{loss_mean} norm:{norm_mean} "
            )

        del optimizer

    def apply_layer_norms(self, block):
        for n, m in block.named_modules():
            if isinstance(m, tuple(_LLMC_LN_TYPES_)):
                m.weight = m.tmp_weight
                del m.tmp_weight
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias = m.tmp_bias
                    del m.tmp_bias
                m.use_tmp_parameter = False

    def register_tweak_parameters(self, block):
        params_dict = {}
        module = FakeQuantLinear
        params_dict["a_qdq"] = self.a_qdq if not self.w_only else None
        params_dict["w_qdq"] = self.w_qdq
        self.model.replace_module_block(module, block, self.block_idx, params_dict)

        llmc_ln_module = _MODEL_LN_TYPES_PAIRS_[self.config["model"]["type"]]
        self.model.replace_module_block(llmc_ln_module, block, self.block_idx, {})

        for n, m in block.named_modules():
            if isinstance(m, tuple(_LLMC_LN_TYPES_)):
                m.register_parameter("tmp_weight", nn.Parameter(m.weight))
                if hasattr(m, "bias") and m.bias is not None:
                    m.register_parameter("tmp_bias", nn.Parameter(m.bias))
                m.use_tmp_parameter = True

    def get_tweak_parameters(self, block):
        params = []
        for n, m in block.named_modules():
            if isinstance(m, tuple(_LLMC_LN_TYPES_)):
                params.append(m.tmp_weight)
                if hasattr(m, "tmp_bias"):
                    params.append(m.tmp_bias)
        return iter(params)

    def adjust_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.ntweak_lr * (
                1 + self.gamma * (self.block_idx / len(self.blocks))
            )

    def deploy(self, quant_format):
        super().deploy(quant_format)
        self.model.convert_dtype(self.model_dtype)

    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)
