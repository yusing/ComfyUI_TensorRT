from .converter import Tensor2RTConvertor
import os

os.environ["CUDA_MODULE_LOADING"] = "1"
NODE_CLASS_MAPPINGS = {"Tensor2RTConvertor": Tensor2RTConvertor}
NODE_DISPLAY_NAME_MAPPINGS = {"Tensor2RTConvertor": "To TensorRT"}
