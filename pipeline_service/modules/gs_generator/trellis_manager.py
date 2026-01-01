from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional
import io

import torch
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        logger.warning("Downloading models from HuggingFace - this may take 5-15 minutes on first run (model files are several GB)")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        start_time = time.time()
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        load_time = time.time() - start_time
        logger.info(f"Models loaded in {load_time:.2f}s. Moving to GPU...")
        
        self.pipeline.cuda()
        total_time = time.time() - start_time
        logger.success(f"Trellis pipeline ready. Total startup time: {total_time:.2f}s")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        trellis_request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")
        images = trellis_request.images
        images = [image.convert("RGB") for image in images]

        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {images[0].size}")

        params = self.default_params.overrided(trellis_request.params)
        start = time.time()
        try:
            if len(images) == 1:
                outputs = self.pipeline.run(
                    images[0],
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                        
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                )
            else:
                outputs = self.pipeline.run_multi_image(
                    images,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,

                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                    mode="multidiffusion",
                )

            generation_time = time.time() - start
            gaussian = outputs["gaussian"][0]

            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            logger.success(f"Trellis finished generation in {generation_time:.2f}s.")
            return result
        finally:
            if buffer:
                buffer.close()

