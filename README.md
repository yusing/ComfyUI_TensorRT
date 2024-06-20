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

- No issues on repository to report this stuff better
- Loras don't work:
![image](https://github.com/phr00t/ComfyUI_TensorRT/assets/5983470/320b724a-8b68-4348-8631-020b61790dba)
- No requirements.txt file
- Weight streaminging doesn't work (and loads in lowvram mode)
![image](https://github.com/phr00t/ComfyUI_TensorRT/assets/5983470/1176a6da-d179-4f12-a011-66714ad146b3)
- AnimateDiff doesn't work
![image](https://github.com/phr00t/ComfyUI_TensorRT/assets/5983470/a66babe5-6dca-4794-bec1-3d87a2c17630)
