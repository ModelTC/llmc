from transformers import AutoModelForCausalLM, AutoConfig
from loguru import logger
from tqdm import tqdm

import gc
import os
import torch
import functools
import argparse
import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import torch.nn as nn

from llmc.data import BaseTokenizer, BaseDataset
from llmc.utils.registry_factory import MODEL_REGISTRY
from llmc.models import *
from llmc.utils import seed_all, check_config, mkdirs
from llmc.compression.quantization import Quantizer
from llmc.compression.quantization import FakeQuantLinear


def draw(save_path, save_name, X, Y1, Y2):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X, Y1)
    ax.plot(X, Y2)
    plt.xlabel("channel")
    plt.ylabel("value")
    plt.title(save_name)
    fig.savefig(f"{save_path}/{save_name}.jpg")
    plt.close(fig)
    plt.cla()


def analysis_block_cosine(res, t_res, args):
    cosine_sim = nn.CosineSimilarity()

    for name in res:
        oups = res[name]
        t_oups = t_res[name]

        layer_cosine_dict = {}
        for j in range(oups.shape[0]):
            cos = cosine_sim(oups[j].float().view(1, -1), t_oups[j].float().view(1, -1))

            if name not in layer_cosine_dict:
                layer_cosine_dict[name] = []

            layer_cosine_dict[name].append(cos.item())

        for name in layer_cosine_dict:
            cos_values = layer_cosine_dict[name]
            min_cos = min(cos_values)
            avg_cos = sum(cos_values) / len(cos_values)
            logger.info(name)
            logger.info(f"min_cos : {min_cos}")
            logger.info(f"avg_cos : {avg_cos}")


def analysis_block_outlier(res, t_res, args):
    for name in res:
        tensor = res[name].mean(dim=0)
        abs_max = tensor.abs().amax(dim=0).detach().cpu().numpy()
        min_val = tensor.amin(dim=0).detach().cpu().numpy()
        max_val = tensor.amax(dim=0).detach().cpu().numpy()

        logger.info(f"The original model {name}'s abs_max:")
        logger.info(abs_max.max())

        t_tensor = t_res[name].mean(dim=0)
        t_abs_max = t_tensor.abs().amax(dim=0).detach().cpu().numpy()
        t_min_val = t_tensor.amin(dim=0).detach().cpu().numpy()
        t_max_val = t_tensor.amax(dim=0).detach().cpu().numpy()

        logger.info(f"The transformed model {name}'s abs_max:")
        logger.info(t_abs_max.max())

        if args.draw:
            save_outlier_path = os.path.join(args.save_path, "outlier")
            save_t_outlier_path = os.path.join(args.save_path, "t_outlier")

            if not os.path.exists(args.save_path):
                mkdirs(save_outlier_path)
                mkdirs(save_t_outlier_path)

            draw(
                save_path=save_outlier_path,
                save_name=name,
                X=range(tensor.shape[-1]),
                Y1=min_val,
                Y2=max_val,
            )

            draw(
                save_path=save_t_outlier_path,
                save_name=name,
                X=range(t_tensor.shape[-1]),
                Y1=t_min_val,
                Y2=t_max_val,
            )


def register_hook(block, idx, args):
    hooks = []
    for name, m in block.named_modules():
        if not args.cosine:
            if isinstance(m, torch.nn.Linear):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            stat_input_hook, name=name, idx=idx, args=args
                        )
                    )
                )
        else:
            if isinstance(m, (torch.nn.Linear, FakeQuantLinear)):
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            stat_output_hook, name=name, idx=idx, args=args
                        )
                    )
                )

    return hooks


def stat_input_hook(m, x, y, name, idx, args):
    if isinstance(x, tuple):
        x = x[0]

    layer_name = f"block_{idx}.{name}"
    if t:
        t_res[layer_name] = x
    else:
        res[layer_name] = x


def stat_output_hook(m, x, y, name, idx, args):
    if isinstance(y, tuple):
        y = y[0]
    layer_name = f"block_{idx}.{name}"
    if t:
        t_res[layer_name] = y
    else:
        res[layer_name] = y


def block_forward(block, input_data, input_kwargs):
    output = []

    for i in range(len(input_data)):
        input_data[i] = input_data[i].to(
            device=next(block.parameters()).device,
            dtype=next(block.parameters()).dtype,
        )
        if "attention_mask" in input_kwargs[i]:
            input_kwargs[i]["attention_mask"] = input_kwargs[i]["attention_mask"].cuda()
        with torch.no_grad():
            out = block(input_data[i], **input_kwargs[i])[0]
            output.append(out)
    return output


class analysis_quanter(Quantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)

    def fake_quant_weight_dynamic(self, module, args={}):
        weight = module.weight
        if "int_indices" in args:
            if self.granularity == "per_group":
                assert len(args["int_indices"]) % self.group_size == 0
            q_weight = weight[:, args["int_indices"]]
            fp_weight = weight[:, args["fp_indices"]]

        elif "dim" in args and "ic" in args["dim"]:
            q_weight = weight.T
        else:
            q_weight = weight

        if "current_bit" in args:
            org_bit = self.bit
            self.bit = args["current_bit"]

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        q_weight, scales, zeros, max_int, min_int = self.get_tensor_qparams(
            q_weight, args
        )

        q_weight = self.quant_dequant(q_weight, scales, zeros, max_int, min_int)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if "current_bit" in args:
            self.bit = org_bit

        if "int_indices" in args:
            mix_weight = torch.zeros_like(weight)
            mix_weight[:, args["int_indices"]] = q_weight
            mix_weight[:, args["fp_indices"]] = fp_weight
            return mix_weight

        elif "dim" in args and "ic" in args["dim"]:
            q_weight = q_weight.T

        return q_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--bs", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preproc", type=str, default="general")
    parser.add_argument("--save_path", type=str, default="./save")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--t_model_path", type=str)
    parser.add_argument("--torch_dtype", type=str, default="auto")

    parser.add_argument("--w_only", action="store_true")
    parser.add_argument("--wbit", type=int, default=4)
    parser.add_argument("--wsym", action="store_true")
    parser.add_argument("--wgra", type=str, default="per_group")
    parser.add_argument("--group_size", type=int, default=128)

    parser.add_argument("--abit", type=int, default=4)
    parser.add_argument("--asym", action="store_true")
    parser.add_argument("--agra", type=str, default="per_token")

    args = parser.parse_args()
    logger.info(f"args : {args}")

    calib_cfg = {
        "name": args.dataset_name,
        "download": False,
        "path": args.data_path,
        "n_samples": args.n_samples,
        "bs": args.bs,
        "seq_len": args.seq_len,
        "preproc": args.preproc,
        "seed": args.seed,
    }

    model_config = {
        "type": args.model_type,
        "path": args.model_path,
        "torch_dtype": args.torch_dtype,
    }

    model = MODEL_REGISTRY[args.model_type](args.model_path, args.torch_dtype)
    t_model = MODEL_REGISTRY[args.model_type](args.t_model_path, args.torch_dtype)

    logger.info(model)

    tokenizer = BaseTokenizer(args.model_path)
    dataset = BaseDataset(tokenizer.get_tokenizer(), calib_cfg)

    calib_data = dataset.get_calib_dataset()

    model.collect_first_block_input(calib_data)
    t_model.collect_first_block_input(calib_data)

    fp_inps = model.get_first_block_input()
    t_fp_inps = t_model.get_first_block_input()

    res = {}
    t_res = {}

    if args.cosine:
        wquanter = analysis_quanter(
            bit=args.wbit,
            symmetric=args.wsym,
            granularity=args.wgra,
            group_size=args.group_size,
        )

        if not args.w_only:
            aquanter = Quantizer(
                bit=args.abit, symmetric=args.asym, granularity=args.agra
            )

        params_dict = {}
        params_dict["w_qdq"] = wquanter.fake_quant_weight_dynamic
        params_dict["a_qdq"] = None if args.w_only else aquanter.fake_quant_act_dynamic
        t_model.replace_module_all(FakeQuantLinear, params_dict)

    with torch.no_grad():
        for i in tqdm(range(len(model.blocks))):
            block = model.blocks[i]
            t_block = t_model.blocks[i]
            block.cuda()
            t_block.cuda()

            hooks = register_hook(block, i, args)
            t = False
            fp_inps["data"] = block_forward(block, fp_inps["data"], fp_inps["kwargs"])

            t_hooks = register_hook(t_block, i, args)
            t = True
            t_fp_inps["data"] = block_forward(
                t_block, t_fp_inps["data"], t_fp_inps["kwargs"]
            )

            block.cpu()

            t_block.cpu()

            for h in hooks:
                h.remove()

            for t_h in t_hooks:
                t_h.remove()

            if args.cosine:
                analysis_block_cosine(res, t_res, args)
            else:
                analysis_block_outlier(res, t_res, args)
            res.clear()
            t_res.clear()

            gc.collect()
            torch.cuda.empty_cache()