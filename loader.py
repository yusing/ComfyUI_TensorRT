import gc
from typing import Optional
import cuda.cudart
import torch

import comfy

import tensorrt as trt

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import comfy.utils

from .lora_stacker import StackedLora
from .utils import deserialize
from .engine_info import EngineInfo

trt.init_libnvinfer_plugins(None, "")

runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
runtime.engine_host_code_allowed = True


def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16


def dtype_from_str(dtype_str):
    if dtype_str == "torch.float16":
        return torch.float16
    elif dtype_str == "torch.float32":
        return torch.float32
    elif dtype_str == "torch.int32":
        return torch.int32
    elif dtype_str == "torch.bfloat16":
        return torch.bfloat16


class EngineLoader(trt.IStreamReader):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def __enter__(self):
        self.br = open(self.path, "rb")
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.br.close()

    def read(self, size: int):
        return self.br.read(size)


# https://github.com/comfyanonymous/ComfyUI_TensorRT/blob/master/tensorrt_loader.py
class Unet:

    def __init__(
        self,
        engine: trt.ICudaEngine,
        engine_info: EngineInfo,
    ) -> None:
        self.engine = engine
        self.engine.weight_streaming_budget_v2 = (
            4 << 30
        )  # TODO: add an option to customize this
        self.context = self.engine.create_execution_context()
        self.dtype = dtype_from_str(engine_info.dtype)
        self.input_names = ["x", "timesteps", "context"]
        if engine_info.y_dim > 0:
            self.input_names.append("y")
        self.input_types = dict(
            map(
                lambda name: (
                    name,
                    trt_datatype_to_torch(self.engine.get_tensor_dtype(name)),
                ),
                self.input_names,
            )
        )
        self.forward = self.__call__

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(
        self,
        x,
        timesteps,
        context,
        y=None,
        control=None,
        transformer_options=None,
        **kwargs,
    ):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        # opt_batch = dims[1][0]
        max_batch = dims[2][0]

        # Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            model_inputs_converted[k] = model_inputs[k].to(self.input_types[k])

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(
            out_shape,
            device=x.device,
            dtype=trt_datatype_to_torch(
                self.engine.get_tensor_dtype(output_binding_name)
            ),
        )
        model_inputs_converted[output_binding_name] = out

        cuda.cudart.cudaSetDevice(x.device.index)
        streams = []
        for i in range(curr_split_batch):
            err, stream = cuda.cudart.cudaStreamCreate()
            if err != cuda.cudart.cudaError_t.cudaSuccess:
                for s in streams:
                    cuda.cudart.cudaStreamDestroy(s)
                raise RuntimeError(f"cudaStreamCreateWithFlags failed with error {err}")
            streams.append(stream)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(
                    k, x[(x.shape[0] // curr_split_batch) * i :].data_ptr()
                )
            self.context.execute_async_v3(stream_handle=streams[i])
        # (err,) = cuda.cudart.cudaDeviceSynchronize()
        for s in streams:
            cuda.cudart.cudaStreamDestroy(s)
        # if err != cuda.cudart.cudaError_t.cudaSuccess:
        #     raise RuntimeError(f"cudaStreamSynchronize failed with error {err}")
        return out

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}


class Engine:
    def __init__(self, cuda_eng, eng_info: EngineInfo):
        self.model = cuda_eng
        self.load_device = comfy.model_management.get_torch_device()
        self.offload_device = self.load_device
        self.current_device = self.load_device
        self.model_options = {"transformer_options": {}}
        self.info = eng_info

    def memory_required(self, *args, **kwargs):
        return 0

    def model_size(self):
        return 0

    def model_dtype(self):
        return dtype_from_str(self.info.dtype)

    def model_patches_to(self, _):
        pass

    def clone(self):
        return self

    def is_clone(self, _):
        return False

    def add_patches(self, *args, **kwargs):
        return []

    def get_model_object(self, name):
        return comfy.utils.get_attr(self.model, name)

    def model_state_dict(self, filter_prefix=None):
        return {}

    def patch_model(self, device_to=None, patch_weights=True):
        pass

    def patch_model_lowvram(
        self, device_to=None, lowvram_model_memory=0, force_patch_weights=False
    ):
        pass

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        pass


# TODO: cuda graph https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimize-performance
def load_engine(
    engine_path: str,
    engine_info_path: str,
    onnx_path: str,
    lora_stack: Optional[StackedLora],
):
    if lora_stack is not None:
        del model
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

    # TODO: lora

    engine_info = EngineInfo.load(engine_info_path)
    conf_init = deserialize(engine_info.model_config_init)
    if engine_info.y_dim > 0:
        conf = conf_init({"adm_in_channels": engine_info.y_dim})
    else:
        conf = conf_init({})
    conf.unet_config = conf_init.unet_config.copy()  # copy from original class
    conf.unet_config["disable_unet_model_creation"] = True

    model_init = deserialize(engine_info.model_init)
    eng = model_init(conf)

    # ?: reduce host memory usage by streamed deserialization
    with EngineLoader(engine_path) as loader:
        cuda_eng = runtime.deserialize_cuda_engine(loader)
        if cuda_eng is None:
            raise Exception("Failed to load engine")
    gc.collect()

    eng.diffusion_model = Unet(cuda_eng, engine_info)

    return Engine(eng, engine_info)
