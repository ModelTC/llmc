import argparse
from importlib.metadata import version

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def func(model_path):
    model_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True
    )
    print(f'model_config : {model_config}')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
    )
    print()
    print(f'model : {model}')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    print()
    print(f'tokenizer : {tokenizer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    print(f"torch : {version('torch')}")
    print(f"transformers : {version('transformers')}")
    print(f"tokenizers : {version('tokenizers')}")
    print(f"huggingface-hub : {version('huggingface-hub')}")
    print(f"datasets : {version('datasets')}")

    func(args.model)
