import comfy
import folder_paths
import torch
import os

from dfloat11 import DFloat11Model, compress_model

from .dfloat11_custom import DFloat11FluxDiffusersModel

from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux

MODEL_TO_PATTERN_DICT = {
    "Flux": {
        r"double_blocks\.\d+": (
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
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
            "modulation.lin",
        ),
    },
    "FluxSchnell": {
        r"double_blocks\.\d+": (
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
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
            "modulation.lin",
        ),
    },
    "Chroma": {
        r"distilled_guidance_layer\.layers\.\d+": (
            "in_layer",
            "out_layer"        
        ),
        r"double_blocks\.\d+": (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ),
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
        ),
    },
    "ChromaRadiance": {
        r"distilled_guidance_layer\.layers\.\d+": (
            "in_layer",
            "out_layer"        
        ),
        r"double_blocks\.\d+": (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ),
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
        ),
        r"nerf_blocks\.\d+": (
            "param_generator", 
        )
    }
}

class DFloat11ModelLoader:
    """
    A custom node to load a DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

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
            pattern_dict=MODEL_TO_PATTERN_DICT[type(model_config).__name__],
            bfloat16_model=model.diffusion_model,
            device=offload_device,
        )

        return (
            comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device),
        )

class DFloat11DiffusersModelLoader:
    """
    A custom node to load a diffusers-native DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (["Flux",],)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_dfloat11_model"
    CATEGORY = "DFloat11"

    def load_dfloat11_model(self, dfloat11_model_name, model_type):
        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        
        # state_dict = convert_diffusers_to_comfyui_flux(comfy.utils.load_torch_file(dfloat11_model_path))
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)
        
        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        unet_config = {
            'image_model': 'flux', 
            'in_channels': 16, 
            'patch_size': 2, 
            'out_channels': 16, 
            'vec_in_dim': 768, 
            'context_in_dim': 4096, 
            'hidden_size': 3072, 
            'mlp_ratio': 4.0, 
            'num_heads': 24, 
            'depth': 19, 
            'depth_single_blocks': 38, 
            'axes_dim': [16, 56, 56], 
            'theta': 10000, 
            'qkv_bias': True, 
            'guidance_embed': True
        }
        
        unet_config["guidance_embed"] = "time_text_embed.guidance_embedder.linear_1.weight" in state_dict
        
        model_config = comfy.supported_models.Flux(unet_config)
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(state_dict, "")
        model = model.to(offload_device)

        DFloat11FluxDiffusersModel.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[model_type],
            bfloat16_model=model.diffusion_model,
            device=offload_device,
        )

        return (
            comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device),
        )

class DFloat11ModelCompressor:
    """
    A custom node to compress a DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bfloat16_model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (list(MODEL_TO_PATTERN_DICT.keys()),)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_bfloat16_model"
    CATEGORY = "DFloat11"

    def load_bfloat16_model(self, bfloat16_model_name, model_type):
        bfloat16_model_path = folder_paths.get_full_path_or_raise("diffusion_models", bfloat16_model_name)

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model = comfy.sd.load_diffusion_model(bfloat16_model_path, {"dtype" : torch.bfloat16})

        save_path = f"{os.path.splitext(bfloat16_model_path)[0]}-DF11"
        
        compress_model(
            model=model.model.diffusion_model,
            pattern_dict=MODEL_TO_PATTERN_DICT[model_type],
            save_path= save_path,
            save_single_file= True,
            check_correctness= True,
            block_range= (0, 100),
        )

        return (save_path,)



NODE_CLASS_MAPPINGS = {
    "DFloat11ModelLoader": DFloat11ModelLoader,
    "DFloat11DiffusersModelLoader" : DFloat11DiffusersModelLoader,
    "DFloat11ModelCompressor" : DFloat11ModelCompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DFloat11ModelLoader": "DFloat11 Model Loader",
    "DFloat11DiffusersModelLoader" : "DFloat11 diffusers-native Model Loader",
    "DFloat11ModelCompressor" : "DFloat11 Model Compressor",
}

