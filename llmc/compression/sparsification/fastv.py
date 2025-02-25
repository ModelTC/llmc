import functools

import torch

from llmc.utils.registry_factory import ALGO_REGISTRY

from ..blockwise_optimization import BlockwiseOpt
from .attn_utils import _update_causal_mask
from .base_blockwise_sparsification import BaseBlockwiseSparsification


@ALGO_REGISTRY
class FastV(BaseBlockwiseSparsification):
    def __init__(self, model, sparsity_config, input, padding_mask, config):
        BlockwiseOpt.__init__(self, model, sparsity_config, input, padding_mask, config)
        self.add_sparse_config()
        self.register_hooks()

    def add_sparse_config(self):
        special_config = self.sparsity_config.get('special', {})
        self.pruning_loc = special_config['pruning_loc']
        special_config['image_token_start_index'] = \
            self.model.pruning_config['image_token_start_index']
        special_config['image_token_length'] = \
            self.model.pruning_config['image_token_length']
        special_config['attn_scores'] = None

        self.model.model.parameters = special_config

    @torch.no_grad()
    def block_opt(self, block, *opt_kwargs):
        pass

    def register_hooks(self):

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        def store_attention_hook(m, x, layer_outputs, pruning_pars):
            layer_attention = layer_outputs[1]
            pruning_pars['attn_scores'] = layer_attention

        def fastv_pruning_hook(module, args, kwargs, pruning_pars):

            rate = pruning_pars['rate']
            image_token_start_index = pruning_pars['image_token_start_index']
            image_token_length = pruning_pars['image_token_length']

            hidden_states = args[0]
            causal_mask = kwargs['attention_mask']
            cache_position = kwargs['cache_position']

            device = hidden_states.device
            # last_layer_attention = layer_outputs[1]
            last_layer_attention = pruning_pars['attn_scores']
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
            pruning_pars['attention_mask'] = causal_mask
            pruning_pars['cache_position'] = cache_position[:new_seq_length]
            pruning_pars['position_ids'] = position_ids
            pruning_pars['position_embeddings'] = None

            return (hidden_states,), kwargs

        def read_parameter_hook(module, args, kwargs, pruning_pars):
            kwargs['attention_mask'] = pruning_pars['attention_mask']
            kwargs['cache_position'] = pruning_pars['cache_position']
            kwargs['position_ids'] = pruning_pars['position_ids']
            kwargs['position_embeddings'] = pruning_pars['position_embeddings']

            return args, kwargs

        self.blocks[self.pruning_loc - 1].register_forward_pre_hook(
            update_output_attentions_hook,
            with_kwargs=True
        )

        self.blocks[self.pruning_loc - 1].register_forward_hook(
            functools.partial(store_attention_hook, pruning_pars=self.model.model.parameters),
        )

        self.blocks[self.pruning_loc].register_forward_pre_hook(
            functools.partial(fastv_pruning_hook, pruning_pars=self.model.model.parameters),
            with_kwargs=True
        )

        for idx in range(self.pruning_loc + 1, len(self.blocks)):
            self.blocks[idx].register_forward_pre_hook(
                functools.partial(read_parameter_hook, pruning_pars=self.model.model.parameters),
                with_kwargs=True
            )
