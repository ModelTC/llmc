import json


def update_autoawq_quant_config(
    config,
    save_quant_path,
):

    if config.quant.weight.granularity == 'per_group':
        group_size = config.quant.weight.group_size
    else:
        group_size = -1

    quant_config = {
        'bits': config.quant.weight.bit,
        'group_size': group_size,
        'modules_to_not_convert': None,
        'quant_method': 'awq',
        'version': config.quant.weight.pack_version.split('_')[0],
        'zero_point': not config.quant.weight.symmetric
    }

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_autoawq = json.load(file)
    if 'quantization_config' in config_autoawq:
        del config_autoawq['quantization_config']
    config_autoawq['quantization_config'] = quant_config
    with open(config_file, 'w') as file:
        json.dump(config_autoawq, file, indent=4)
