# Extended ComfyUI Plugin for DFloat11

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
