from loguru import logger
from collections import defaultdict
import functools
import torch
import gc
from .sparse import Sparser
from ..blockwise_optimization import BlockwiseOpt


class BaseBlockwiseSparsification(BlockwiseOpt):
    def __init__(self, model, sparsity_config, input, config):
        super().__init__(model, sparsity_config, input, config)
        self.set_sparsity_config()

    def block_init(self, block):
        pass

    def set_sparsity_config(self):
        if (
            "sparsity_out" in self.sparsity_config
            and self.sparsity_config["sparsity_out"]
        ):
            self.sparsity_out = True
        else:
            self.sparsity_out = False
        logger.info(f"use sparsity_out {self.sparsity_out}")
        self.sparser = Sparser(**self.sparsity_config["weight"])

    def block_forward(self, block, input_data=None):
        output = []
        if input_data is None:
            input_data = self.input["data"]

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if (
                "attention_mask" in self.input["kwargs"][i]
                and self.input["kwargs"][i]["attention_mask"] is not None
            ):
                self.input["kwargs"][i]["attention_mask"] = self.input["kwargs"][i][
                    "attention_mask"
                ].cuda()
            with torch.no_grad():
                out = block(input_data[i], **self.input["kwargs"][i])[0]
                output.append(out)
        return output

    def block_opt(self, block, idx):
        block = block.cuda()
        named_linears = self.model.get_block_linears(block)
        # logger.info(f"named_linears: {named_linears}")
        input_feat = defaultdict(list)
        handles = []
        self.block_init(block)

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(
                        self.cache_input_hook, name=name, feat_dict=input_feat
                    )
                )
            )

        if not self.sparsity_out:
            self.input["data"] = self.block_forward(block)
        else:
            self.block_forward(block)
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

        self.block_transform(block, input_feat, idx, self.input["kwargs"])

        if self.sparsity_out:
            self.input["data"] = self.block_forward(block)

        block = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    def block_transform(self, block, input_feat, idx, block_kwargs):
        logger.info(f"Start transform the {idx+1}-th block")
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            if not self.filter_subset(subset):
                continue
            # logger.info(f"subset: {subset}")
            prev_op = subset["prev_op"]
            layers_dict = subset["layers"]
            input_name = subset["input"][0]
            inspect_module = subset["inspect"]
            inspect_has_kwargs = subset["has_kwargs"]
            subset_kwargs = block_kwargs if inspect_has_kwargs else {}
            self.subset_transform(
                layers_dict,
                input_feat,
                prev_op,
                input_name,
                inspect_module,
                subset_kwargs,
                idx,
            )
        logger.info(f"End transform the {idx+1}-th block")

    def filter_subset(self, subset):
        return True

    # todo
    @torch.no_grad()
    def deploy(self):
        logger.info(f"-- deploy_sparsity_model start --")
        logger.info(f"sparsity_config : {self.sparsity_config}")

        # self.model.replace_module_all(module, params_dict)
        logger.info(f"-- deploy_sparsity_model done --")
