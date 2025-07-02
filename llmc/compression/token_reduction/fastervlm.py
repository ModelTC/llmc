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

        special_config['image_attentions_list'] = []

        self.pruning_paras = special_config

    def register_reduction_modules(self):

        def update_output_attentions_hook(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs

        def clear_attentions_hook(m, x, pruning_paras):
            pruning_paras['image_attentions_list'].clear()

        def store_attention_hook(m, x, image_forward_outs, pruning_paras):
            image_attentions = image_forward_outs.attentions[pruning_paras['select_layer']]
            if pruning_paras['select_feature'] in ('default', 'patch'):
                image_attention = image_attentions[:, :, 0, 1:]
            elif pruning_paras['select_feature'] in ('full', 'cls_patch'):
                image_attention = image_attentions
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature}')
            pruning_paras['image_attentions_list'].append(image_attention.to(x[0].dtype))

        def update_attentions_hook(m, x, outs, pruning_paras):
            if len(pruning_paras['image_attentions_list']) == 1:
                pruning_paras['image_attentions'] = pruning_paras['image_attentions_list'][0]
            else:
                pruning_paras['image_attentions'] = pruning_paras['image_attentions_list']

        def pruning_hook(module, args, kwargs, pruning_paras):

            # for llavahf bs 1
            if 'image_attentions' not in pruning_paras:
                pruning_paras['image_attentions'] = pruning_paras['image_attentions_list'][0]

            image_features = args[0]
            image_attentions = pruning_paras['image_attentions']

            B, N, C = image_features.shape
            visual_token_num = self.visual_token_num  # T

            # prune visual tokens by attention scores
            image_attentions = image_attentions.mean(dim=1)  # (B, N)
            token_indices = torch.topk(image_attentions, k=visual_token_num, dim=1)[1]  # (B, T)

            # generate index mask
            index_masks = torch.zeros(
                B, N,
                dtype=torch.bool,
                device=image_features.device
            )  # (B, N)
            index_masks.scatter_(1, token_indices, True)  # (B, N)

            pruning_paras['index_masks'] = index_masks

            return (image_features,), kwargs

        def get_image_mask_hook(module, args, kwargs, pruning_paras):
            pruning_paras['image_masks'] = (
                kwargs['input_ids'] == pruning_paras['image_token_index']
            )  # (B, len)

        def prepare_inputs_for_llm_hook(module, args, kwargs, pruning_paras):

            # Only batch size 1 is currently supported.
            inputs_embeds = kwargs['inputs_embeds']
            image_mask = pruning_paras['image_masks'][0]
            index_mask = pruning_paras['index_masks'][0]

            B, L = inputs_embeds.shape[:2]
            device = inputs_embeds.device

            visual_indexs = torch.arange(L, device=device)[image_mask]
            keep_visual_indexs = visual_indexs[index_mask]

            non_visual_indexs = torch.arange(L, device=device)[~image_mask]

            keep_indexs = torch.cat([non_visual_indexs, keep_visual_indexs]).sort().values

            new_inputs_embeds = kwargs['inputs_embeds'][:, keep_indexs, :]
            new_attention_mask = kwargs['attention_mask'][:, keep_indexs]
            new_position_ids = kwargs['position_ids'][:, keep_indexs]
            new_cache_position = kwargs['cache_position'][keep_indexs]

            kwargs['inputs_embeds'] = new_inputs_embeds
            kwargs['attention_mask'] = new_attention_mask
            kwargs['position_ids'] = new_position_ids
            kwargs['cache_position'] = new_cache_position

            return args, kwargs

        def prepare_inputs_hook(module, inputs, outputs, pruning_paras):

            image_features = outputs
            index_masks = pruning_paras['index_masks']
            # image_attentions = pruning_paras['image_attentions']
            new_image_features = []
            for image_feature, index_mask in zip(image_features, index_masks):
                image_feature = image_feature[index_mask]
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0)

            outputs = image_features
            pruning_paras['image_features_shape'] = image_features[0].shape[0]

            return outputs

        if self.model.__class__.__name__ == 'LlavaHf':
            self.model.vision_model.register_forward_pre_hook(
                update_output_attentions_hook,
                with_kwargs=True
            )

            self.model.vision_model.register_forward_hook(
                functools.partial(store_attention_hook, pruning_paras=self.pruning_paras),
            )
        elif self.model.__class__.__name__ == 'Llava':
            self.model.vision_model.register_forward_pre_hook(
                functools.partial(clear_attentions_hook, pruning_paras=self.pruning_paras),
            )

            self.model.vision_model.register_forward_hook(
                functools.partial(update_attentions_hook, pruning_paras=self.pruning_paras),
            )

            self.model.vision_model.vision_tower.register_forward_pre_hook(
                update_output_attentions_hook,
                with_kwargs=True
            )

            self.model.vision_model.vision_tower.register_forward_hook(
                functools.partial(store_attention_hook, pruning_paras=self.pruning_paras),
            )

        self.model.vision_projector.register_forward_pre_hook(
            functools.partial(pruning_hook, pruning_paras=self.pruning_paras),
            with_kwargs=True
        )

        if self.model.__class__.__name__ == 'LlavaHf':
            self.model.vlm_model.register_forward_pre_hook(
                functools.partial(get_image_mask_hook, pruning_paras=self.pruning_paras),
                with_kwargs=True
            )
            self.model.model.model.register_forward_pre_hook(
                functools.partial(prepare_inputs_for_llm_hook, pruning_paras=self.pruning_paras),
                with_kwargs=True
            )
        elif self.model.__class__.__name__ == 'Llava':
            self.model.vision_projector.register_forward_hook(
                functools.partial(prepare_inputs_hook, pruning_paras=self.pruning_paras),
            )
