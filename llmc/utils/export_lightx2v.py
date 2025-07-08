import json


def update_lightx2v_quant_config(save_quant_path):

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_lightx2v = json.load(file)
    config_lightx2v['quant_method'] = 'advanced_ptq'
    with open(config_file, 'w') as file:
        json.dump(config_lightx2v, file, indent=4)
