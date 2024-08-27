import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmc.utils import mkdirs


def attention_visualization(model, tokenizer, args):
    layer_index = args.layer_idx
    head_index = args.head_idx
    input_text = args.input_text
    save_img_path = args.save_img_path

    inputs = tokenizer(input_text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        attentions = outputs.attentions

    attention_map = attentions[layer_index]
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    if args.all_heads:
        head_idxs = list(range(attention_map.shape[1]))
    else:
        head_idxs = [head_index]

    for idx in head_idxs:
        attn_map = attention_map[0, idx].cpu().numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
        plt.title(f'Attention Map - Layer {layer_index + 1} Head {idx + 1}')
        plt.xlabel('Input Tokens')
        plt.ylabel('Output Tokens')

        save_name = f'layers_{layer_index + 1}_heads_{idx + 1}'
        plt.savefig(f'{save_img_path}/{save_name}.jpg')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--all_heads', action='store_true')
    parser.add_argument('--layer_idx', type=int, default=0)
    parser.add_argument('--head_idx', type=int, default=0)
    parser.add_argument('--save_img_path', type=str, default='./save')

    args = parser.parse_args()

    mkdirs(args.save_img_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 output_attentions=True)

    attention_visualization(model, tokenizer, args)
