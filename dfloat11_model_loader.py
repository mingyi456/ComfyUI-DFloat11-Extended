import comfy
import folder_paths
import torch

from dfloat11 import DFloat11Model

class DFloat11ModelLoader:
    """
    A custom node to load a DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

    MODEL_TO_PATTERN_DICT = {
        "Flux": {
            "double_blocks\.\d+": (
                "img_mod.lin",
                "img_attn.qkv",
                "img_attn.proj",
                "img_mlp.0",
                "img_mlp.2",
                "txt_mod.lin",
                "txt_attn.qkv",
                "txt_attn.proj",
                "txt_mlp.0",
                "txt_mlp.2",
            ),
            "single_blocks\.\d+": (
                "linear1",
                "linear2",
                "modulation.lin",
            ),
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_dfloat11_model"
    CATEGORY = "DFloat11"

    def load_dfloat11_model(self, dfloat11_model_name):
        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        sd = comfy.utils.load_torch_file(dfloat11_model_path)

        if not any(k.endswith("encoded_exponent") for k in sd.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_config = comfy.sd.model_detection.model_config_from_unet(sd, "")
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(sd, "")
        model = model.to(offload_device)

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=DFloat11ModelLoader.MODEL_TO_PATTERN_DICT[type(model_config).__name__],
            bfloat16_model=model.diffusion_model,
            device=offload_device,
        )

        return (
            comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device),
        )


NODE_CLASS_MAPPINGS = {
    "DFloat11ModelLoader": DFloat11ModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11ModelLoader": "DFloat11 Model Loader",
}
