import gc
import os

import torch
from diffusers.utils import export_to_video
from loguru import logger

from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY

from .eval_base import BaseEval


class VideoGenerateEval(BaseEval):

    def __init__(self, model, config):
        super().__init__(model, config)
        self.output_video_path = self.eval_cfg.get('output_video_path', None)
        assert self.output_video_path is not None
        os.makedirs(self.output_video_path, exist_ok=True)
        self.target_height = self.eval_cfg.get('target_height', 480)
        self.target_width = self.eval_cfg.get('target_width', 832)
        self.num_frames = self.eval_cfg.get('num_frames', 81)
        self.guidance_scale = self.eval_cfg.get('guidance_scale', 5.0)
        self.fps = self.eval_cfg.get('fps', 15)

    @torch.no_grad()
    def eval(self, model_llmc, eval_pos):
        seed_all(self.config.base.seed + int(os.environ['RANK']))
        model_llmc.Pipeline.to('cuda')
        eval_res = self.eval_func(
            model_llmc,
            self.testenc,
            self.eval_dataset_bs,
            eval_pos,
        )

        model_llmc.Pipeline.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        return eval_res

    @torch.no_grad()
    def eval_func(self, model, testenc, bs, eval_pos):
        assert bs == 1, 'Only support eval bs=1'

        for i, data in enumerate(testenc):
            output = model.Pipeline(
                prompt=data['prompt'],
                negative_prompt=data['negative_prompt'],
                height=self.target_height,
                width=self.target_width,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
            ).frames[0]
            export_to_video(
                output,
                os.path.join(self.output_video_path, f'{eval_pos}_output_{i}.mp4'),
                fps=self.fps,
            )

        return None
