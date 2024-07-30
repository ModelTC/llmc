def get_wquantizer(
    block_idx, layer_name, mix_bits_map, quantizer_mix_bits, wquantizer_default
):
    mix_bits_map_this_block = mix_bits_map[block_idx]
    for layer_name_substring in mix_bits_map_this_block:
        if layer_name_substring in layer_name:
            quantizer_mix_bits_this_layer = quantizer_mix_bits[
                mix_bits_map_this_block[layer_name_substring]
            ]
            if quantizer_mix_bits_this_layer["do_quant"]:
                assert "wquantizer" in quantizer_mix_bits_this_layer
                return quantizer_mix_bits_this_layer["wquantizer"]
            else:
                return None  # This layer do not quant.
    return wquantizer_default


def get_aquantizer(
    block_idx, layer_name, mix_bits_map, quantizer_mix_bits, aquantizer_default
):
    mix_bits_map_this_block = mix_bits_map[block_idx]
    for layer_name_substring in mix_bits_map_this_block:
        if layer_name_substring in layer_name:
            quantizer_mix_bits_this_layer = quantizer_mix_bits[
                mix_bits_map_this_block[layer_name_substring]
            ]
            if quantizer_mix_bits_this_layer["do_quant"]:
                assert "aquantizer" in quantizer_mix_bits_this_layer
                return quantizer_mix_bits_this_layer["aquantizer"]
            else:
                return None  # This layer do not quant.
    return aquantizer_default


def check_do_quant(block_idx, layer_name, mix_bits_map, quantizer_mix_bits):
    mix_bits_map_this_block = mix_bits_map[block_idx]
    for layer_name_substring in mix_bits_map_this_block:
        if layer_name_substring in layer_name:
            quantizer_mix_bits_this_layer = quantizer_mix_bits[
                mix_bits_map_this_block[layer_name_substring]
            ]
            return quantizer_mix_bits_this_layer["do_quant"]
    return True


def check_w_only(
    block_idx, layer_name, mix_bits_map, quantizer_mix_bits, default_w_only
):
    mix_bits_map_this_block = mix_bits_map[block_idx]
    for layer_name_substring in mix_bits_map_this_block:
        if layer_name_substring in layer_name:
            quantizer_mix_bits_this_layer = quantizer_mix_bits[
                mix_bits_map_this_block[layer_name_substring]
            ]
            return quantizer_mix_bits_this_layer["w_only_mix_bits"]
    return default_w_only
