# Extended ComfyUI Plugin for DFloat11

<b>Currently supported model architectures:
- Flux.1-dev and schnell
- Chroma
- Chroma-Radiance
- SDXL (UNET component only, requires the `--bf16-unet` command-line flag)
- Cosmos Predict 2 Text2Image
- Lumina Image 2.0
- Z-Image</b>

<b>Currently supported features:
- Compatibility node for loading `diffusers`-native Flux DF11 models (using the "DFloat11 diffusers-native Model Loader" node)
- Cpu offloading options (available under the "DFloat11 Model Loader (Advanced)" node)
- Experimental LoRA support for Chroma</b>

Thanks to @tonyzhang617 for implementing the base DF11 compression and inference code. Unfortunately, it seems that the original developer is rather sporadic in his efforts to maintain the codebase and add features, so I decided to fork the repo and attempt to support it myself. 

Currently, my fork adds support for more model types (compared to just Flux-dev based models in the original repo), and I plan to support more base architectures in future. Also, I have added a node ("DFloat11 diffusers-native Model Loader") for loading existing Flux DF11 models that are native to the `diffusers` library by converting them on-the-fly, which reduces the need to provide DF11 compression for both ComfyUI and `diffusers` separately with the same model. Sadly, it appears that my current runtime conversion code causes ComfyUI to use up slighly more VRAM as compared to using a ComfyUI-native DF11 model, so I will have to support both nodes concurrently for now. In any case, even with the extra VRAM usage, VRAM consumption still stays below 24 GB (technically even the full Flux BF16 weights can also run with 24GB of VRAM, just that using literally 100% of the available VRAM affects system responsiveness). Another drawback of the on-the-fly conversion process is that there is a noticeable speed penalty (~1.30 it/s vs ~1.55 it/s).

Finally, the "DFloat11 Model Compressor" node allows users to generate their own ComfyUI-native DF11 compressions for Flux and Chroma models. The compression process uses only the CPU for compression, while the GPU is only used for verification, so less than 4GB of VRAM is used in the process. This means almost anyone should be able to create their own compressions, provided they have sufficient system RAM. Compressing a 12B model (i.e. Flux-based models) takes up almost 48 GB of RAM. Currently, the compression is single-threaded, but I guess waiting an hour or two for the compression process should be fine.

Check out my HuggingFace profile here: https://huggingface.co/mingyi456. I have uploaded a few DF11 models that are natively compatible with the original DF11 custom node, while the rest are `diffusers`-native and require my own added node to work with ComfyUI. Feel free to create an [issue](https://github.com/mingyi456/ComfyUI-DFloat11-Extended/issues/new/choose) to request other models for compression as well (either for `diffusers` or ComfyUI), although models that use architectures I am unfamiliar with might be more difficult.

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dfloat11?period=total\&units=INTERNATIONAL_SYSTEM\&left_color=BLACK\&right_color=GREEN\&left_text=downloads)](https://pepy.tech/projects/dfloat11)
[![arXiv](https://img.shields.io/badge/arXiv-2504.11651-b31b1b.svg)](https://arxiv.org/abs/2504.11651)
[![Hugging Face](https://img.shields.io/badge/Model-%F0%9F%A4%97-yellow.svg)](https://huggingface.co/DFloat11)

This repository provides the **ComfyUI plugin for DFloat11 models**.

DFloat11 reduces model size by more than **30%** while producing **bit-for-bit identical outputs** to the original. Unlike quantization techniques which trade quality for size, DFloat11 is a **lossless compression method**, preserving model output quality fully while supporting efficient inference.

Currently, only **FLUX.1 models** are supported. Support for additional models is planned. Please feel free to [open an issue](https://github.com/LeanModels/ComfyUI-DFloat11/issues) and let us know which ones you'd like to see next.

---

## Features

* 🚀 **Fully Lossless** – 100% identical outputs to the original model
* 📦 **>30% smaller model size** – lower VRAM requirements than the original model
* ⚡ **Compatible with ComfyUI** – drop-in support with custom nodes
* 🔧 **GPU-accelerated inference** – optimized for CUDA 12.1+

---

## Installation

### Requirements

* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed
* NVIDIA GPU with **CUDA 12.1+**

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the DFloat11 custom nodes in ComfyUI:

   ```bash
   cd <ComfyUI_installation_path>/custom_nodes
   git clone https://github.com/LeanModels/ComfyUI-DFloat11.git
   ```

---

## Usage

1. Once installed, the DFloat11 nodes show up under the `DFloat11` folder in the *Node Library*.
2. Download a DFloat11 model for ComfyUI from [Hugging Face](https://huggingface.co/DFloat11) and place it under `<ComfyUI_installation_path>/models/diffusion_models`.
3. Drag and drop a `*.png` or `*.json` file from [workflows](https://github.com/LeanModels/ComfyUI-DFloat11/tree/master/workflows) into ComfyUI to load the workflow.
4. (Optional) Use the `DFloat11 Model Loader` node to load the model in `*.safetensors` format, which acts as a drop-in replacement for the `Load Diffusion Model` node.

---

## Resources

* 📖 [DFloat11 Paper (arXiv)](https://arxiv.org/abs/2504.11651)
* 🤗 [DFloat11 Models on Hugging Face](https://huggingface.co/DFloat11)

---

## Contributing

Contributions are welcome!

* Open an issue to request new model support
* Submit pull requests for bug fixes or improvements

[![Contributors](https://contrib.rocks/image?repo=LeanModels/ComfyUI-DFloat11)](https://github.com/LeanModels/ComfyUI-DFloat11/graphs/contributors)
