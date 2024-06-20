import os
import folder_paths

ENGINE_EXT = ".engine"
ENGINE_INFO_EXT = ".json"
ONNX_EXT = ".pb"

folder_paths.folder_names_and_paths["trt"] = (
    [os.path.join(folder_paths.models_dir, "trt")],
    [ENGINE_EXT],
)
folder_paths.folder_names_and_paths["trt_cache"] = (
    [os.path.join(folder_paths.models_dir, "trt_cache")],
    [],
)

CACHE_PATH: str = folder_paths.get_folder_paths("trt_cache")[0]
OUTPUT_PATH: str = folder_paths.get_folder_paths("trt")[0]
TIMING_CACHE_FILE: str = os.path.join(CACHE_PATH, "timing_cache.trt")
