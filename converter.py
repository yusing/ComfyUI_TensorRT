from os import path
import gc
import os
from typing import Any, Optional
import torch
import tensorrt as trt
from tqdm import tqdm
import comfy.model_management
import comfy.model_base
import comfy.model_patcher
import comfy.sd
import comfy.supported_models
import comfy.supported_models_base
import folder_paths
import nodes

from .lora_stacker import StackedLora
from .engine_info import EngineInfo
from .loader import Engine, load_engine
from .paths import (
    CACHE_PATH,
    ENGINE_EXT,
    ONNX_EXT,
    OUTPUT_PATH,
    ENGINE_INFO_EXT,
    TIMING_CACHE_FILE,
)
from .utils import half, double, serialize

INPUT_NAMES = ["x", "timesteps", "context", "y"]
OUTPUT_NAMES = ["o"]

MIN_BATCHES = 1
MAX_BATCHES = 50
MIN_HEIGHT = 512
MAX_HEIGHT = 4096
MIN_WIDTH = 512
MAX_WIDTH = 4096
DEFAULT_OPT_LEVEL = 3
MIN_OPT_LEVEL = 1
MAX_OPT_LEVEL = 5


class ComfyDiffusionModel(torch.nn.Module):
    def __init__(self, model, opts) -> None:
        super().__init__()
        self.diffusion_model = model
        self.transformer_options = opts

    def forward(self, x, timesteps, context, y=None, *_, **__):
        return self.diffusion_model(
            x,
            timesteps,
            context,
            y=y,
            transformer_options=self.transformer_options,
        )


class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build,
            # so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False


class Tensor2RTConvertor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "lora_stack": ("TRT_LORA_STACK",),
            },
            "required": {
                "model_path": (folder_paths.get_filename_list("checkpoints"),),
                "batch_size": (
                    "INT",
                    {
                        "default": MIN_BATCHES,
                        "min": MIN_BATCHES,
                        "max": MAX_BATCHES,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": MIN_HEIGHT,
                        "min": MIN_HEIGHT,
                        "max": MAX_HEIGHT,
                        "step": 64,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": MIN_WIDTH,
                        "min": MIN_WIDTH,
                        "max": MAX_WIDTH,
                        "step": 64,
                    },
                ),
                "optimization_level": (
                    "INT",
                    {
                        "default": DEFAULT_OPT_LEVEL,
                        "min": MIN_OPT_LEVEL,
                        "max": MAX_OPT_LEVEL,
                        "step": 1,
                    },
                ),
                "force_rebuild": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LATENT", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = (
        "model",
        "clip",
        "vae",
        "latent",
        "model_name",
        "batch_size",
        "height",
        "width",
    )
    FUNCTION = "convert"
    CATEGORY = "TorchTensorRT/Load Checkpoint"

    LatentImage = dict[str, torch.Tensor]
    ReturnType = tuple[Engine, Any, Any, LatentImage, str, int, int, int]

    # Sets up the builder to use the timing cache file, and creates it if it does not already exist
    def _setup_timing_cache(self, config: trt.IBuilderConfig):
        buffer = b""
        if os.path.exists(TIMING_CACHE_FILE):
            with open(TIMING_CACHE_FILE, mode="rb") as timing_cache_file:
                buffer = timing_cache_file.read()
            print("Read {} bytes from timing cache.".format(len(buffer)))
        else:
            print("No timing cache found; Initializing a new one.")
        timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)

    # Saves the config's timing cache to file
    def _save_timing_cache(self, config: trt.IBuilderConfig):
        timing_cache: trt.ITimingCache = config.get_timing_cache()
        with open(TIMING_CACHE_FILE, "wb") as timing_cache_file:
            timing_cache_file.write(memoryview(timing_cache.serialize()))

    def load_checkpoint(
        self,
        path: str,
        model=True,
        clip=True,
        vae=True,
    ):
        return comfy.sd.load_checkpoint_guess_config(
            folder_paths.get_full_path("checkpoints", path),
            output_model=model,
            output_vae=vae,
            output_clip=clip,
            output_clipvision=False,
        )

    def empty_latent_image(self, width: int, height: int, batch_size=1):
        return nodes.EmptyLatentImage().generate(width, height, batch_size=batch_size)[
            0
        ]

    def __init__(self) -> None:
        self.last_path: str = ""
        self.models: tuple[Engine, Any, Any]

    def convert(
        self,
        model_path: str,
        batch_size: int,
        height: int,
        width: int,
        optimization_level: int = DEFAULT_OPT_LEVEL,
        force_rebuild: bool = False,
        lora_stack: Optional[StackedLora] = None,
    ) -> ReturnType:
        model_name = path.basename(model_path).split(".")[0]
        engine_path = path.join(OUTPUT_PATH, model_name + ENGINE_EXT)
        engine_info_path = engine_path + ENGINE_INFO_EXT
        onnx_path = path.join(CACHE_PATH, model_name, model_name + ONNX_EXT)

        def results() -> Tensor2RTConvertor.ReturnType:
            return (
                *self.models,
                self.empty_latent_image(width, height, batch_size),
                model_name,
                batch_size,
                height,
                width,
            )

        if path == self.last_path:
            return results()

        def load():
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            gc.collect()

            engine = load_engine(engine_path, engine_info_path, onnx_path, lora_stack)
            _, clip, vae, _ = self.load_checkpoint(model_path, model=False)
            self.last_path = model_path
            self.models = (engine, clip, vae)
            return results()

        if (
            not force_rebuild
            and os.path.exists(engine_path)
            and os.path.exists(engine_info_path)
        ):
            return load()

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        model, *_ = self.load_checkpoint(model_path, clip=False, vae=False)
        # NOTE: this line is needed when not running in --high-vram
        comfy.model_management.load_models_gpu([model], force_patch_weights=True)

        dtype = model.model.diffusion_model.dtype or torch.float16
        device = comfy.model_management.get_torch_device()

        engine_info = EngineInfo(
            min_batch=half(batch_size, MIN_BATCHES, MAX_BATCHES),
            opt_batch=batch_size,
            max_batch=double(batch_size, MIN_BATCHES, MAX_BATCHES),
            min_height=half(height, MIN_HEIGHT, MAX_HEIGHT),
            opt_height=height,
            max_height=double(height, MIN_HEIGHT, MAX_HEIGHT),
            min_width=half(width, MIN_WIDTH, MAX_WIDTH),
            opt_width=width,
            max_width=double(width, MIN_WIDTH, MAX_WIDTH),
            min_context_len=1,
            opt_context_len=77,
            max_context_len=77,
            in_channels=int(model.model.model_config.unet_config.get("in_channels")),
            context_dim=0,
            y_dim=int(model.model.adm_channels),
            dtype=str(dtype),
            model_config_init=serialize(model.model.model_config.__class__),
            model_init=serialize(model.model.__class__),
        )

        dynamic_axes = {
            "x": {0: "batch", 2: "height", 3: "width"},
            "timesteps": {0: "batch"},
            "context": {0: "batch", 1: "num_embeds"},
        }
        if engine_info.y_dim > 0:
            dynamic_axes["y"] = {0: "batch"}

        assert model.model.diffusion_model is not None
        context_dim = model.model.model_config.unet_config.get("context_dim", None)
        if context_dim is None:  # SD3
            context_embedder_config = model.model.model_config.unet_config.get(
                "context_embedder_config", None
            )
            if context_embedder_config is not None:
                context_dim = context_embedder_config.get("params", {}).get(
                    "in_features", None
                )
                engine_info.opt_context_len = 154
                engine_info.max_context_len = 154
        if context_dim is None:
            raise Exception("Unsupported model, could not determine context dim")
        engine_info.context_dim = int(context_dim)

        min_shapes, opt_shapes, max_shapes = (
            engine_info.min_shapes(),
            engine_info.opt_shapes(),
            engine_info.max_shapes(),
        )

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        if force_rebuild or not os.path.exists(onnx_path):
            torch.onnx.export(
                ComfyDiffusionModel(
                    model.model.diffusion_model,
                    model.model_options["transformer_options"].copy(),
                ),
                tuple(
                    map(
                        lambda s: torch.zeros(s, device=device, dtype=dtype),
                        opt_shapes,
                    )
                ),
                f=onnx_path,
                input_names=INPUT_NAMES,
                output_names=OUTPUT_NAMES,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                # export_params=False,
                verbose=False,
            )

        del model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        if force_rebuild or not os.path.exists(engine_path):
            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(
                (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                | (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
            )
            parser = trt.OnnxParser(network, logger)
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

            success = parser.parse_from_file(onnx_path)
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            if not success:
                raise Exception("failed to load ONNX file")

            print("engine_info", engine_info)

            config = builder.create_builder_config()
            config.max_aux_streams = 0
            config.builder_optimization_level = optimization_level

            # NOTE: disabled because trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED is enabled
            # config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(
                trt.BuilderFlag.WEIGHT_STREAMING
            )  # save some VRAM by allowing weights to be streamed in
            config.set_flag(trt.BuilderFlag.DIRECT_IO)
            # config.set_flag(trt.BuilderFlag.STRIP_PLAN)
            # config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)
            config.progress_monitor = TQDMProgressMonitor()
            self._setup_timing_cache(config)

            profile = builder.create_optimization_profile()
            for i, name in enumerate(INPUT_NAMES):
                profile.set_shape(name, min_shapes[i], opt_shapes[i], max_shapes[i])

            config.add_optimization_profile(profile)

            del parser, profile
            gc.collect()

            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise Exception("failed to build serialized engine")
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            self._save_timing_cache(config)

            del builder, network, config, serialized_engine
            comfy.model_management.soft_empty_cache()
            gc.collect()

        engine_info.dump(engine_info_path)

        return load()
