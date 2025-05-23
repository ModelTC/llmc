from functools import wraps
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer


def prefill_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # for the decoding stage
        if len(args) > 1:
            input_args = args[1]
            if hasattr(input_args[0], 'shape') and input_args[0].shape[1] == 1:
                return None
        return func(*args, **kwargs)
    return wrapper


def prefill_wrapper_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # for the decoding stage
        if len(args) > 1:
            input_args = args[2]['inputs_embeds']
            if hasattr(input_args, 'shape') and input_args.shape[1] == 1:
                return None
        return func(*args, **kwargs)
    return wrapper


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """Copy from the TOME. https://github.com/facebookresearch/ToMe.

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


def make_tome_class(transformer_class):
    class VisionZipTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._info['r'] = parse_r(len(self.vision_model.encoder.layers), self.r)
            # self._info["r"] = self.r

            self._info['size'] = None
            self._info['source'] = None

            return super().forward(*args, **kwdargs)

    return VisionZipTransformer


def apply_info(model, dominant_num, contextual_num):

    VisionZipTransformer = make_tome_class(model.__class__)

    model.__class__ = VisionZipTransformer
    model.r = [0 for i in range(22)] + [1] + [0]

    model._info = {
        'r': [model.r],
        'dominant': dominant_num,
        'contextual': contextual_num,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module.self_attn.k_proj._info = model._info
