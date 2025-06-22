import functools
from types import MethodType

import torch

from llmc.compression.sparsification.attn_utils import _update_causal_mask
from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule
from .utils import prefill_wrapper

IMAGE_TOKEN_INDEX = -200


@TOKEN_REDUCTION_REGISTRY.register('FastV')
class FastV(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):

        self.pruning_loc = self.special_config['pruning_loc']
        self.special_config['image_token_length'] = \
            self.model.pruning_config['image_token_length']
        self.special_config['attn_scores'] = None

        self.pruning_paras = self.special_config

    def register_reduction_modules(self):

        @prefill_wrapper
        def input_hook(module, input_args, pruning_paras):
            input_ids = input_args[0]
            image_token_idxs = (input_ids[0] ==
                                pruning_paras['vision_token_index']).nonzero(as_tuple=True)[0]
            pruning_paras['image_token_start_index'] = image_token_idxs[0].item()

            return input_args

        def make_hook_prepare_inputs_labels_for_multimodal(pruning_paras):
            def hook_prepare_inputs_labels_for_multimodal(
                self,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            ):
                if 'image_token_start_index' not in pruning_paras:
                    token_indices = input_ids[0][attention_mask[0]] == IMAGE_TOKEN_INDEX
                    pruning_paras['image_token_start_index'] = torch.where(token_indices)[0].item()
                return self._original_prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask,
                    past_key_values, labels, images, image_sizes
                )
            return hook_prepare_inputs_labels_for_multimodal

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        def store_attention_hook(m, x, layer_outputs, pruning_paras):
            layer_attention = layer_outputs[1]
            pruning_paras['attn_scores'] = layer_attention

        @prefill_wrapper
        def fastv_pruning_hook(module, args, kwargs, pruning_paras):

            rate = pruning_paras['rate']
            image_token_start_index = pruning_paras['image_token_start_index']
            image_token_length = pruning_paras['image_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']
            cache_position = kwargs['cache_position']

            device = hidden_states.device
            # last_layer_attention = layer_outputs[1]
            last_layer_attention = pruning_paras['attn_scores']
            # compute average attention over different head
            last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
            # generate new attention mask based on the average attention,
            # sample the top ATTENTION_RANK tokens with highest attention
            last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
            # get the attention in image token
            last_layer_attention_avg_last_tok_image = \
                last_layer_attention_avg_last_tok[image_token_start_index:
                                                  image_token_start_index + image_token_length]
            # get the indexes of the top ATTENTION_RANK tokens
            top_attention_rank_index = \
                last_layer_attention_avg_last_tok_image.topk(
                    round(image_token_length * (1 - rate))).indices + image_token_start_index
            # keep index
            keep_indexs = torch.cat(
                (
                    torch.arange(image_token_start_index, device=device),
                    top_attention_rank_index,
                    torch.arange(image_token_start_index + image_token_length,
                                 hidden_states.shape[1], device=device)
                )
            )

            # sort index
            keep_indexs = keep_indexs.sort().values
            # update seq length
            new_seq_length = keep_indexs.shape[0]
            # filter hidden states &

            hidden_states = hidden_states[:, keep_indexs, :]
            # update position ids
            position_ids = keep_indexs.unsqueeze(0)
            # update attention mask
            causal_mask = _update_causal_mask(
                causal_mask, None, hidden_states, 0
            ) if causal_mask is not None else None
            kwargs['attention_mask'] = causal_mask
            kwargs['cache_position'] = cache_position[:new_seq_length]
            kwargs['position_ids'] = position_ids
            kwargs['position_embeddings'] = None
            pruning_paras['attention_mask'] = causal_mask
            pruning_paras['cache_position'] = cache_position[:new_seq_length]
            pruning_paras['position_ids'] = position_ids
            pruning_paras['position_embeddings'] = None

            return (hidden_states,), kwargs

        @prefill_wrapper
        def read_parameter_hook(module, args, kwargs, pruning_paras):
            kwargs['attention_mask'] = pruning_paras['attention_mask']
            kwargs['cache_position'] = pruning_paras['cache_position']
            kwargs['position_ids'] = pruning_paras['position_ids']
            kwargs['position_embeddings'] = pruning_paras['position_embeddings']

            return args, kwargs

        if self.model.__class__.__name__ == 'LlavaHf':
            self.model.embed_tokens.register_forward_pre_hook(
                functools.partial(input_hook, pruning_paras=self.pruning_paras)
            )
        elif self.model.__class__.__name__ == 'Llava':
            hook_fn = make_hook_prepare_inputs_labels_for_multimodal(self.pruning_paras)
            self.model.vlm_model._original_prepare_inputs_labels_for_multimodal = (
                self.model.vlm_model.prepare_inputs_labels_for_multimodal
            )
            self.model.vlm_model.prepare_inputs_labels_for_multimodal = MethodType(
                hook_fn, self.model.vlm_model
            )

        self.blocks[self.pruning_loc - 1].register_forward_pre_hook(
            update_output_attentions_hook,
            with_kwargs=True
        )

        self.blocks[self.pruning_loc - 1].register_forward_hook(
            functools.partial(store_attention_hook, pruning_paras=self.pruning_paras),
        )

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(fastv_pruning_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )

        for idx in range(self.pruning_loc + 1, len(self.blocks)):
            self.blocks[idx].register_forward_pre_hook(
                functools.partial(read_parameter_hook, pruning_paras=self.pruning_paras),
                with_kwargs=True
            )
