

import sys

autoawq_path = '/path/to/AutoAWQ'
sys.path.append(autoawq_path)

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

model_path = '/path/to/save_for_autoawq_awq_w4/autoawq_quant_model'

tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

model = AutoAWQForCausalLM.from_quantized(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
)


prompt_text = 'The president of the United States is '
inputs = tokenizer(prompt_text, return_tensors='pt').to('cuda')

outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=100,
    streamer=streamer,
    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
)
