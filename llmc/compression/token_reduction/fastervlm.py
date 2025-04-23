import functools

import torch

from llmc.compression.sparsification.attn_utils import _update_causal_mask
from llmc.utils.registry_factory import TOKEN_REDUCTION_REGISTRY

from .token_reduction_module import TokenReductionModule


@TOKEN_REDUCTION_REGISTRY.register('FasterVLM')
class FasterVLM(TokenReductionModule):
    def __init__(self, config, model, blocks):
        super().__init__(config, model, blocks)
        self.add_sparse_config()
        self.register_reduction_modules()

    def add_sparse_config(self):
        special_config = self.config.get('special', {})
        self.visual_token_num = round(
            self.model.pruning_config['image_token_length'] * (1 - special_config['rate'])
        )
        special_config['select_layer'] = self.model.pruning_config['select_layer']
        special_config['select_feature'] = self.model.pruning_config['select_feature']
        special_config['image_token_index'] = self.model.pruning_config['image_token_index']

        self.model.model.parameters = special_config

    def register_reduction_modules(self):

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        def store_attention_hook(m, x, image_forward_outs, pruning_pars):
            image_attentions = image_forward_outs.attentions[pruning_pars['select_layer']]
            if pruning_pars['select_feature'] == 'default':  # patch
                image_attentions = image_attentions[:, :, 0, 1:]
            elif pruning_pars['select_feature'] == 'full':
                image_attentions = image_attentions
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            pruning_pars['image_attentions'] = image_attentions

        def pruning_hook(module, args, kwargs, pruning_pars):

            image_features = args[0]
            image_attentions = pruning_pars['image_attentions']

            # image_attentions = image_attentions.max(dim=1)[0] # (B, N) = (1, 576)
            image_attentions = image_attentions.mean(dim=1)  # (B, N) = (1, 576)

            B, N = image_features.shape[:2]
            visual_token_num = self.visual_token_num  # T

            # prune visual tokens by random scores
            # token_weights = torch.rand(B, N, device=image_features.device)  # (B, N)
            # token_indices = torch.topk(token_weights, k=visual_token_num, dim=1)[1]   # (B, T)
            # token_indices = torch.sort(token_indices, dim=1)[0]  # (B, T)

            # prune visual tokens by attention scores
            token_indices = torch.topk(image_attentions, k=visual_token_num, dim=1)[1]  # (B, T)
            token_indices = torch.sort(token_indices, dim=1)[0]  # (B, T)

            # generate index mask
            index_mask = torch.zeros(B, N, dtype=torch.bool, device=image_features.device)  # (B, N)
            index_mask.scatter_(1, token_indices, True)  # (B, N)

            pruning_pars['index_mask'] = index_mask
            pruning_pars['image_attentions'] = image_attentions

            return (image_features,), kwargs

        def get_image_mask_hook(module, args, kwargs, pruning_pars):
            pruning_pars['image_mask'] = (
                kwargs['input_ids'] == pruning_pars['image_token_index']
            )  # (B, len)

        def prepare_inputs_for_llm_hook(module, args, kwargs, pruning_pars):

            # Only batch size 1 is currently supported.
            inputs_embeds = kwargs['inputs_embeds']
            image_mask = pruning_pars['image_mask'][0]
            index_mask = pruning_pars['index_mask'][0]

            B, L = inputs_embeds.shape[:2]
            device = inputs_embeds.device

            visual_indexs = torch.arange(L, device=device)[image_mask]
            keep_visual_indexs = visual_indexs[index_mask]

            non_visual_indexs = torch.arange(L, device=device)[~image_mask]

            keep_indexs = torch.cat([non_visual_indexs, keep_visual_indexs]).sort().values

            new_inputs_embeds = kwargs['inputs_embeds'][:, keep_indexs, :]

            new_attention_mask = torch.ones(
                new_inputs_embeds.shape[:2],
                dtype=kwargs['attention_mask'].dtype, device=device
            )
            new_position_ids = torch.arange(new_inputs_embeds.shape[1], device=device).unsqueeze(0)
            new_cache_position = kwargs['cache_position'][keep_indexs]

            kwargs['inputs_embeds'] = new_inputs_embeds
            kwargs['attention_mask'] = new_attention_mask
            kwargs['position_ids'] = new_position_ids
            kwargs['cache_position'] = new_cache_position

            return args, kwargs

        self.model.vision_model.register_forward_pre_hook(
            update_output_attentions_hook,
            with_kwargs=True
        )

        self.model.vision_model.register_forward_hook(
            functools.partial(store_attention_hook, pruning_pars=self.model.model.parameters),
        )

        self.model.vision_projector.register_forward_pre_hook(
            functools.partial(pruning_hook, pruning_pars=self.model.model.parameters),
            with_kwargs=True
        )

        self.model.vlm_model.register_forward_pre_hook(
            functools.partial(get_image_mask_hook, pruning_pars=self.model.model.parameters),
            with_kwargs=True
        )

        self.model.model.register_forward_pre_hook(
            functools.partial(
                prepare_inputs_for_llm_hook, pruning_pars=self.model.model.parameters
            ),
            with_kwargs=True
        )
