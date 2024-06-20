# TensorRT Node for ComfyUI

Similar to [https://github.com/comfyanonymous/ComfyUI_TensorRT], but builds 2-3x faster and uses only 2-3GB of VRAM (default optimization level).

## Differences

- No need to choose model type, auto detected
- No need to separate build and output process
- Customizable optimization level (higher=faster run time but slower and more memory intense on build)
- Memory usage tweaks, (i.e. streamed weight loading, etc.)
- **Compatible to LORAs, FreeU, HyperTile, etc.**
- Have all necessary params as output (i.e. `model_name`, `batch_size`, etc.), making it compatible with *Image Saver*

## Current Issue

Only tested on SDXL
