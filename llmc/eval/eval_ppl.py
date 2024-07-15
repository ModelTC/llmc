import torch
import torch.nn as nn
from loguru import logger
import gc
from datasets import load_dataset, load_from_disk
from concurrent.futures import ThreadPoolExecutor


class PerplexityEval:
    def __init__(self, tokenizer, eval_cfg):
        self.tokenizer = tokenizer
        # eval_cfg
        logger.info(f"eval_cfg : {eval_cfg}")
        self.dataset = eval_cfg["name"]
        assert self.dataset in [
            "wikitext2",
            "c4",
            "ptb",
        ], "Ppl eval only support wikitext2, c4, ptb dataset now."
        self.seq_len = eval_cfg["seq_len"]
        self.bs = eval_cfg["bs"]
        self.path = eval_cfg.get("path", None)
        self.download = eval_cfg["download"]
        self.inference_per_block = eval_cfg.get("inference_per_block", False)
        self.testenc = self.build_data()

    @torch.no_grad()
    def build_data(self):
        # load data
        if self.download:
            if self.dataset == "wikitext2":
                testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            elif self.dataset == "c4":
                testdata = load_dataset(
                    "allenai/c4",
                    data_files={
                        "validation": "en/c4-validation.00000-of-00008.json.gz"
                    },
                    split="validation",
                )
            elif self.dataset == "ptb":
                testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        else:
            assert self.path, "Please set path in eval_cfg."
            testdata = load_from_disk(self.path)

        # encode data
        if self.dataset == "wikitext2":
            testenc = self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        elif self.dataset == "c4":
            testenc = self.tokenizer(
                " ".join(testdata[:1100]["text"]), return_tensors="pt"
            )
            testenc.input_ids = testenc.input_ids[:, : (256 * self.seq_len)]
        elif self.dataset == "ptb":
            testenc = self.tokenizer(
                " ".join(testdata["sentence"]), return_tensors="pt"
            )
        return testenc

    @torch.no_grad()
    def eval(self, model_llmc):
        model = model_llmc.get_model()
        if self.inference_per_block:
            handles = []
            for layer in model_llmc.get_blocks():
                handles.append(layer.register_forward_pre_hook(self.forward_pre_hook))
            for layer in model_llmc.get_blocks():
                handles.append(layer.register_forward_hook(self.forward_hook))
            for layer in model_llmc.get_layers_except_blocks():
                layer.cuda()
        else:
            model.cuda()

        model.eval()
        ppl = self.eval_ppl_func(model, self.testenc, self.seq_len, self.bs)
        if self.inference_per_block:
            for h in handles:
                h.remove()
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        return ppl

    @torch.no_grad()
    def forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def forward_hook(self, m, x, y):
        with ThreadPoolExecutor() as executor:
            executor.submit(self.load_layer_to_cpu, m)

    @torch.no_grad()
    def load_layer_to_cpu(self, m):
        m.cpu()

    @torch.no_grad()
    def eval_ppl_func(self, model, testenc, seq_len, bs):
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seq_len

        nlls = []

        # Loop through each batch
        for i in range(0, nsamples, bs):
            logger.info(f"index : {(i + 1) // bs}/{nsamples // bs}")
            # Calculate end index
            j = min(i + bs, nsamples)

            # Prepare inputs and move to gpu
            inputs = testenc[:, (i * seq_len) : (j * seq_len)].cuda()
            inputs = inputs.reshape(j - i, seq_len)

            # Forward pass through the model
            lm_logits = model(inputs).logits

            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )

            # Calculate negative log likelihood
            neg_log_likelihood = loss.float() * seq_len * (j - i)

            # Append to list of negative log likelihoods
            nlls.append(neg_log_likelihood)

        # Compute perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))

        # Empty CUDA cache to save memory
        testenc.cpu()
        torch.cuda.empty_cache()

        return ppl.item()


if __name__ == "__main__":
    import sys

    sys.path.append("../../")
    import argparse
    from llmc.models import Llama
    from llmc.data import BaseTokenizer
    from llmc.utils.registry_factory import MODEL_REGISTRY

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = BaseTokenizer(args.model_path)
    model = MODEL_REGISTRY[args.model_type](args.model_path, "auto")

    # Llama2-70B config example
    eval_cfg = {
        "name": "wikitext2",
        "seq_len": 2048,
        "bs": 20,
        "download": False,
        "path": "data_path",
        "inference_per_block": True,
    }
    ppl_eval = PerplexityEval(tokenizer.get_tokenizer(), eval_cfg)

    ppl = ppl_eval.eval(model)
    logger.info(f"ppl : {ppl}")
