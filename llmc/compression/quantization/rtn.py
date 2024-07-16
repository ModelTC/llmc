from .base_blockwise_quantization import BaseBlockwiseQuantization
from llmc.utils.registry_factory import ALGO_REGISTRY
from loguru import logger
import torch


@ALGO_REGISTRY
class RTN(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input=None, config=None):
        super().__init__(model, quant_config, input, config)
        if quant_config.get("act", False) and quant_config["act"].get("static", False):
            logger.info("Activation quant is static. Calibration is required.")
            self.act_static = True
        else:
            self.act_static = False

    @torch.no_grad()
    def block_opt(self, *opt_kwargs):
        pass

    def a_qdq(self, act, module, aquantizer):
        if self.act_static:
            args = {}
            args["scales"] = (
                module.buf_act_scales if hasattr(module, "buf_act_scales") else None
            )
            args["zeros"] = (
                module.buf_act_zeros if hasattr(module, "buf_act_zeros") else None
            )
            args["max_int"] = (
                module.buf_act_max_int if hasattr(module, "buf_act_max_int") else None
            )
            args["min_int"] = (
                module.buf_act_min_int if hasattr(module, "buf_act_min_int") else None
            )
            return self.aquantizer.fake_quant_act_static(act, args)
        else:
            return self.aquantizer.fake_quant_act_dynamic(act)

    def get_act_qparams(self, layers_dict, act_tensors):
        avg_min_val, avg_max_val = None, None
        for act_tensor in act_tensors:
            act_tensor = self.aquantizer.reshape_tensor(act_tensor)
            tensor_range = self.aquantizer.get_tensor_range(act_tensor)
            min_val, max_val = tensor_range[0], tensor_range[1]
            if min_val is None:
                avg_min_val = None
            else:
                if avg_min_val is None:
                    avg_min_val = min_val / len(act_tensors)
                else:
                    avg_min_val += min_val / len(act_tensors)
            if max_val is None:
                avg_max_val = None
            else:
                if avg_max_val is None:
                    avg_max_val = max_val / len(act_tensors)
                else:
                    avg_max_val += max_val / len(act_tensors)
        scales, zeros, max_int, min_int = self.aquantizer.get_qparams(
            (avg_min_val, avg_max_val), act_tensors[0].device
        )
        for name in layers_dict:
            layers_dict[name].register_buffer("buf_act_scales", scales)
            layers_dict[name].register_buffer("buf_act_zeros", zeros)
            layers_dict[name].register_buffer("buf_act_max_int", max_int)
            layers_dict[name].register_buffer("buf_act_min_int", min_int)
            logger.info(f"{name} act_scales : {scales}")
            logger.info(f"{name} act_zeros : {zeros}")
            logger.info(f"{name} act_max_int : {max_int}")
            logger.info(f"{name} act_min_int : {min_int}")

    @torch.no_grad()
    def subset_transform(
        self,
        layers_dict,
        input_feat,
        prev_op,
        input_name,
        inspect_module,
        subset_kwargs,
    ):
        if not self.act_static:
            logger.info("Activation quant is dynamic. Do nothing here.")
            return
        self.get_act_qparams(layers_dict, input_feat[input_name])
