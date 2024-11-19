import os

# 文件路径
cpu_txt_path = "cpu.txt"


def modify_file(filepath, modifications):
    with open(filepath, "r") as file:
        lines = file.readlines()

    # 应用修改
    new_lines = []
    for line in lines:
        # 替换操作
        for search, replace in modifications["modifications"]:
            if search in line:
                line = line.replace(search, replace)
        new_lines.append(line)

    # 在文件开头插入新内容
    with open(filepath, "w") as file:
        file.writelines(modifications["header"] + new_lines)


def main():
    with open(cpu_txt_path, "r") as file:
        file_paths = file.readlines()

    for file_path in file_paths:
        file_path = file_path.strip()
        if not file_path:
            continue

        if file_path == "../llmc/__main__.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()"
                    ),
                    (
                        "init_process_group(backend='nccl')",
                        "init_process_group(backend='gloo')"
                    ),
                    (
                        "torch.cuda.set_device(int(os.environ['LOCAL_RANK']))",
                        "# torch.cuda.set_device(int(os.environ['LOCAL_RANK']))"
                    )
                ],
            }
        elif file_path == "../llmc/compression/quantization/awq.py":
            modifications = {
                "header": ["n_grid_zbl = 1\n"],
                "modifications": [
                    (
                        "n_grid = 20",
                        "n_grid = n_grid_zbl"
                    ),
                    (
                        "device='cuda'",
                        "device='cpu'"
                    )
                ],
            }
        elif file_path == "../llmc/compression/quantization/gptq.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()"
                    ),
                    (".cuda()", ".to(device_zbl)"),
                    ("torch.device('cuda')", "torch.device('cpu')"),
                    (
                        "torch.cuda.synchronize()",
                        "if use_cuda: torch.cuda.synchronize()"
                    ),
                ],
            }
        elif (
            file_path
            == "../llmc/compression/quantization/base_blockwise_quantization.py"
        ):
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()",
                    ),
                ],
            }
        elif file_path == "../llmc/models/base_model.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        
                        "self.move_embed_to_device('cuda')",
                        "self.move_embed_to_device(device_zbl)",
                    ),
                ],
            }
        elif file_path == "../llmc/eval/eval_base.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()",
                    ),
                ],
            }
        elif file_path == "../llmc/eval/eval_ppl.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                    "nsamples_zbl = 1\n",
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()",
                    ),
                    ("nlls = []", "nlls = []; nsamples = nsamples_zbl"),
                ],
            }
        elif file_path == "../llmc/eval/eval_token_consist.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                    "nsamples_zbl = 1\n",
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()",
                    ),
                    ("for i in range(0, nsamples, bs):", "for i in range(0, 1, 1):"),
                ],
            }
        elif file_path== "../llmc/compression/quantization/auto_clip.py":
            modifications = {
                "header": [
                    'device_zbl = "cpu"\n',
                    'use_cuda = (device_zbl != "cpu")\n',
                ],
                "modifications": [
                    (".cuda()", ".to(device_zbl)"),
                    (
                        "torch.cuda.empty_cache()",
                        "if use_cuda: torch.cuda.empty_cache()",
                    ),
                ],
            }
        else:
            print(f"File {file_path} not recognized or not specified for modification.")
            continue

        # 修改文件
        if os.path.exists(file_path):
            modify_file(file_path, modifications)
            print(f'{file_path} was modefied successfully')
        else:
            print(f"File {file_path} does not exist.")


if __name__ == "__main__":
    main()
