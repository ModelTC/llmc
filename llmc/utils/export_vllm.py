import json


def update_vllm_quant_config(
    model,
    config,
    save_quant_path,
    vllm_quant_method='compressed-tensors',

):
    pack_mode = config.quant.weight.get('pack_mode')
    if pack_mode is not None:
        vllm_quant_format = 'pack-quantized'
    else:
        vllm_quant_format = 'int-quantized'

    if config.quant.weight.granularity == 'per_group':
        group_size = config.quant.weight.group_size
    else:
        group_size = None

    quant_config = {
        'config_groups': {
            'group_0': {
                'targets': ['Linear'],  # Now only support "Linear".
                'input_activations': {
                    'dynamic': True,
                    'group_size': None,   # Don't support activations per-group quant.
                    'num_bits': config.quant.act.bit,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'token',   # Now only support dynamic per-token
                    'symmetric': config.quant.act.symmetric,
                    'type': 'int'
                } if 'act' in config.quant else None,
                'weights': {
                    'dynamic': False,
                    'group_size': group_size,
                    'num_bits': config.quant.weight.bit,
                    'observer': 'minmax',  # Now only support "minmax".
                    'observer_kwargs': {},
                    'strategy': (
                        'group'
                        if config.quant.weight.granularity == 'per_group'
                        else 'channel'
                    ),
                    'symmetric': config.quant.weight.symmetric,
                    'type': 'int',
                },
            }
        },
        'format': vllm_quant_format,
        'ignore': model.skip_layer_name(),
        'quant_method': vllm_quant_method,
    }

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_vllm = json.load(file)
    config_vllm['compression_config'] = quant_config
    with open(config_file, 'w') as file:
        json.dump(config_vllm, file, indent=4)
