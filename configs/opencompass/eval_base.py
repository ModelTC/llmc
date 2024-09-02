from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel


with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets

datasets = [*humaneval_datasets]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='LLMC-OPENCOMPASS',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
