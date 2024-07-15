import torch
from loguru import logger
from abc import abstractmethod, ABCMeta


class BlockwiseOpt(metaclass=ABCMeta):
    def __init__(self, model, quant_config, input, config):
        self.model = model
        self.blocks = model.get_blocks()
        self.quant_config = quant_config
        self.sparsity_config = quant_config
        self.input = input
        self.config = config
        if self.input:
            for i in range(len(input["kwargs"])):
                if "use_cache" in input["kwargs"][i]:
                    input["kwargs"][i].pop("use_cache")
            self.n_samples = 0
            for i in range(len(input["data"])):
                self.n_samples += input["data"][i].shape[0]

    def run_block_loop(self):
        for i in range(len(self.blocks)):
            logger.info(f"\nindex: {i+1}/{len(self.blocks)} \nblock: {self.blocks[i]}")
            self.block_opt(self.blocks[i], i)

        if hasattr(self, "save_scale") and self.save_scale:
            torch.save(self.act_scales, self.scale_path)
        if hasattr(self, "save_clip") and self.save_clip:
            torch.save(self.weight_clips, self.clip_path)

    @abstractmethod
    def block_opt(self, block, idx):
        pass

    def cache_input_hook(self, m, x, y, name, feat_dict):
        x = x[0]
        x = x.detach().cpu()
        feat_dict[name].append(x)
