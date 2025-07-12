import functools
import math
from functools import wraps
from types import MethodType

import torch

from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper


@TOKEN_REDUCTION_REGISTRY.register('DART')
class DART(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config['pruning_loc']
        self.special_config['image_token_length'] = \
            self.model.pruning_config['image_token_length']
        self.special_config['IMAGE_TOKEN_INDEX'] = \
            self.model.pruning_config['IMAGE_TOKEN_INDEX']

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        def input_hook_llava(fn, pruning_paras):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                if len(args) == 0:
                    return fn(*args, **kwargs)
                input_args = args[0]
                if hasattr(input_args[0], 'shape') and input_args[0].shape[0] == 1:
                    return fn(*args, **kwargs)

                input_ids = args[0]
                attention_mask = args[2]
                token_indices = (
                    input_ids[0][attention_mask[0]] == pruning_paras['IMAGE_TOKEN_INDEX']
                )
                pruning_paras['image_token_start_index'] = torch.where(token_indices)[0][0].item()

                outputs = fn(*args, **kwargs)
                return outputs
            return wrapper

        def get_seq_len_hook(module, args, kwargs, pruning_paras):
            if kwargs['input_ids'] is not None:
                pruning_paras['seq_len'] = kwargs['input_ids'].shape[1]
            elif kwargs['inputs_embeds'] is not None:
                pruning_paras['seq_len'] = kwargs['inputs_embeds'].shape[1]
            else:
                raise ValueError('You have to specify either input_ids or inputs_embeds')

        def get_any_states_hook(module, args, kwargs, layer_outs, pruning_paras, layer_idx):
            from transformers.models.llama.modeling_llama import (
                apply_rotary_pos_emb, repeat_kv)
            if len(kwargs['position_ids'][0]) == 1:
                return layer_outs

            hidden_states = kwargs['hidden_states']
            position_embeddings = kwargs['position_embeddings']
            position_ids = kwargs['position_ids']
            past_key_value = layer_outs[2]

            bsz, q_len, _ = hidden_states.size()
            query_states = module.q_proj(hidden_states)
            key_states = module.k_proj(hidden_states)
            value_states = module.v_proj(hidden_states)
            query_states = query_states.view(
                bsz, q_len, module.num_heads, module.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, module.num_key_value_heads, module.head_dim
            ).transpose(1, 2)

            if position_embeddings is None:
                cos, sin = module.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                key_states = past_key_value.key_cache[layer_idx]
                value_states = past_key_value.value_cache[layer_idx]
            key_states = repeat_kv(key_states, module.num_key_value_groups)
            value_states = repeat_kv(value_states, module.num_key_value_groups)

            pruning_paras['any_states'] = (query_states, key_states, value_states)

            return layer_outs

        @prefill_wrapper
        def pruning_hook(module, args, kwargs, pruning_paras, normlayer):

            image_token_start_index = pruning_paras['image_token_start_index']
            image_token_length = pruning_paras['image_token_length']
            any_states = pruning_paras['any_states'][-2]
            seq_length = pruning_paras['seq_len']

            hidden_states = args[0]
            attention_mask = kwargs['attention_mask']
            device = hidden_states.device
            last_layer_state = normlayer(hidden_states)

            # keep index
            retained_image_tokens_index = get_retained_image_token(
                pruning_paras, last_layer_state, any_states)

            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    retained_image_tokens_index,
                    torch.arange(
                        image_token_start_index + image_token_length,
                        seq_length,
                        device=device
                    )
                )
            )
            # sort index
            keep_indexs = keep_indexs.sort().values
            hidden_states = hidden_states[:, keep_indexs, :]
            position_ids = keep_indexs.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask[
                    :, :, :hidden_states.shape[1], :hidden_states.shape[1]
                ]
                kwargs['attention_mask'].resize_as_(attention_mask).copy_(attention_mask.clone())
            kwargs['cache_position'].resize_as_(position_ids.squeeze(0)).copy_(
                position_ids.squeeze(0).clone())
            kwargs['position_ids'].resize_as_(position_ids).copy_(position_ids.clone())

            position_embeddings = kwargs['position_embeddings']
            new_pe0 = position_embeddings[0][:, keep_indexs, :].clone()
            new_pe1 = position_embeddings[1][:, keep_indexs, :].clone()
            position_embeddings[0].resize_as_(new_pe0).copy_(new_pe0)
            position_embeddings[1].resize_as_(new_pe0).copy_(new_pe1)

            return (hidden_states,), kwargs

        hook_fn = input_hook_llava(
            self.model.vlm_model.prepare_inputs_labels_for_multimodal,
            self.pruning_paras
        )
        self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
            hook_fn, self.model.vlm_model
        )

        self.model.model.model.register_forward_pre_hook(
            functools.partial(get_seq_len_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )

        self.blocks[self.pruning_loc - 1].self_attn.register_forward_hook(
            functools.partial(
                get_any_states_hook,
                pruning_paras=self.pruning_paras,
                layer_idx=self.pruning_loc - 1
            ),
            with_kwargs=True
        )

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(
                pruning_hook,
                pruning_paras=self.pruning_paras,
                normlayer=self.model.model.model.norm
            ),
            with_kwargs=True
        )


def get_retained_image_token(pruning_paras, last_layer_state, any_states):
    image_token_start_index = pruning_paras['image_token_start_index']
    image_token_length = pruning_paras['image_token_length']
    MAX_NUM_TRUNCTION = pruning_paras['max_num_trunction']
    pivot_image_token = pruning_paras['pivot_image_token']
    pivot_text_token = pruning_paras['pivot_text_token']
    reduction_ratio = pruning_paras['reduction_ratio']
    TOKEN_TOPK = math.ceil(
        (
            MAX_NUM_TRUNCTION if MAX_NUM_TRUNCTION is not None
            else (image_token_length * (1 - reduction_ratio))
        ) // (pivot_image_token + pivot_text_token))
    device = last_layer_state.device

    any_states = any_states.permute(0, 2, 1, 3)
    any_states = any_states.reshape(any_states.shape[0], any_states.shape[1], -1)

    k_states_image_token = any_states[0][
        image_token_start_index:image_token_start_index + image_token_length, :
    ]
    k_states_query_token = any_states[0][image_token_start_index + image_token_length:, :]

    k_states_image_token_L1_norm = torch.norm(k_states_image_token, p=1, dim=-1)
    k_states_query_token_L1_norm = torch.norm(k_states_query_token, p=1, dim=-1)

    image_indices = (
        k_states_image_token_L1_norm.topk(pivot_image_token).indices
        + image_token_start_index
    ).tolist()
    query_indices = (
        k_states_query_token_L1_norm.topk(pivot_text_token).indices
        + image_token_start_index + image_token_length
    ).tolist()
    indices_set = set(image_indices + query_indices)

    valid_indices = set(
        range(image_token_start_index, image_token_start_index + image_token_length)
    ) - set(image_indices)

    valid_indices_list = list(valid_indices)
    for item in list(indices_set):
        valid_vectors = last_layer_state[0][valid_indices_list, :]
        cos_sim = -torch.nn.functional.cosine_similarity(
            last_layer_state[0][item, :],
            valid_vectors,
            dim=-1
        )
        top_k_indices = cos_sim.topk(TOKEN_TOPK).indices

        top_k_real_indices = [valid_indices_list[i] for i in top_k_indices]
        indices_set.update(top_k_real_indices)

        valid_indices.difference_update(top_k_real_indices)
        valid_indices_list = list(valid_indices)

    indices_set.difference_update(query_indices)

    retained_image_tokens_index = torch.tensor(list(indices_set), device=device)

    return retained_image_tokens_index
