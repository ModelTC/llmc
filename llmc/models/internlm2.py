from .base_model import BaseModel
from llmc.utils.registry_factory import MODEL_REGISTRY


@MODEL_REGISTRY
class InternLM2(BaseModel):
    def __init__(self, model_path, torch_dtype):
        super().__init__(model_path, torch_dtype)

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.tok_embeddings = self.model.model.tok_embeddings

    def find_block_name(self):
        self.block_name_prefix = "model.layers"

    def get_embed_layers(self):
        return [self.tok_embeddings]

    def get_layers_except_blocks(self):
        return [self.tok_embeddings, self.model.model.norm, self.model.output]

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            "attention_norm": block.attention_norm,
            "ffn_norm": block.ffn_norm,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                "layers": {"attention.wqkv": block.attention.wqkv},
                "prev_op": [block.attention_norm],
                "input": ["attention.wqkv"],
                "inspect": block.attention,
                "has_kwargs": True,
            },
            {
                "layers": {"attention.wo": block.attention.wo},
                "prev_op": [block.attention.wqkv],
                "input": ["attention.wo"],
                "inspect": block.attention.wo,
                "has_kwargs": False,
            },
            {
                "layers": {
                    "feed_forward.w3": block.feed_forward.w3,
                    "feed_forward.w1": block.feed_forward.w1,
                },
                "prev_op": [block.ffn_norm],
                "input": ["feed_forward.w1"],
                "inspect": block.feed_forward,
                "has_kwargs": False,
            },
            {
                "layers": {"feed_forward.w2": block.feed_forward.w2},
                "prev_op": [block.feed_forward.w3],
                "input": ["feed_forward.w2"],
                "inspect": block.feed_forward.w2,
                "has_kwargs": False,
            },
        ]
