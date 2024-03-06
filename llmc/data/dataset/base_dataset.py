from abc import ABCMeta
from datasets import load_dataset, load_from_disk
import torch
from loguru import logger
from .specified_preproc import PREPROC_REGISTRY


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, tokenizer, calib_cfg):
        # calib_cfg
        logger.info(f"calib_cfg : {calib_cfg}")
        self.tokenizer = tokenizer
        self.calib_dataset_name = calib_cfg["name"]
        self.download = calib_cfg["download"]
        self.load_from_txt = calib_cfg.get("load_from_txt", False)
        self.calib_dataset_path = calib_cfg.get("path", None)
        self.n_samples = calib_cfg["n_samples"]
        self.calib_bs = calib_cfg["bs"]
        self.seq_len = calib_cfg["seq_len"]
        self.preproc = calib_cfg["preproc"]
        self.seed = calib_cfg["seed"]
        self.dataset_key = {
            "pileval": "text",
            "c4": "text",
            "wikitext2": "text",
            "ptb": "sentence",
        }
        if self.calib_dataset_name in self.dataset_key:
            self.key = self.dataset_key[self.calib_dataset_name]
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.download:
            if self.calib_dataset_name == "pileval":
                self.calib_dataset = load_dataset(
                    "mit-han-lab/pile-val-backup", split="validation"
                )
            elif self.calib_dataset_name == "c4":
                self.calib_dataset = load_dataset(
                    "allenai/c4",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split="train",
                )
            elif self.calib_dataset_name == "wikitext2":
                self.calib_dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split="train"
                )
            elif self.calib_dataset_name == "ptb":
                self.calib_dataset = load_dataset(
                    "ptb_text_only", "penn_treebank", split="train"
                )
            else:
                raise Exception(f"Not support {self.calib_dataset_name} dataset.")
        else:
            if not self.load_from_txt:
                # Need to pre-download the dataset.
                self.calib_dataset = load_from_disk(self.calib_dataset_path)
            else:
                """
                Load dataset from your custom txt file.
                Each line in the txt file represents one input text data.
                """
                assert self.calib_dataset_path.endswith(".txt")
                logger.info(f"calib_dataset_path: {self.calib_dataset_path}")
                with open(self.calib_dataset_path, "r") as fp:
                    lines = fp.readlines()
                self.calib_dataset = []
                for line in lines:
                    self.calib_dataset.append(line.strip())

    def get_calib_samples(self):
        if self.preproc == "general":
            samples = self.general_preproc(
                self.calib_dataset, self.tokenizer, self.n_samples, self.seq_len
            )
        else:
            preproc = PREPROC_REGISTRY[self.preproc]
            samples = preproc(
                self.calib_dataset, self.tokenizer, self.n_samples, self.seq_len
            )
        return samples

    def get_calib_dataset(self):
        samples = self.get_calib_samples()
        calib_samples = []
        if self.calib_bs < 0:
            batch = torch.cat(samples, dim=0)
            calib_samples.append(batch)
        elif self.calib_bs == 1:
            calib_samples = samples
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                batch = torch.cat(batch, dim=0)
                calib_samples.append(batch)
        logger.info(f"len(calib_samples) : {len(calib_samples)}")
        return calib_samples

    def general_preproc(self, calib_dataset, tokenizer, n_samples, seq_len):
        dataset = calib_dataset.shuffle(seed=self.seed)
        samples = []
        n_run = 0
        for data in dataset:
            line = data[self.key]
            trainenc = tokenizer(
                line, return_tensors="pt", max_length=seq_len, truncation=True
            )
            line_encoded = trainenc.input_ids
            if line_encoded.shape[1] < seq_len:
                continue
            samples.append(line_encoded)
            n_run += 1
            if n_run == n_samples:
                break
        return samples
