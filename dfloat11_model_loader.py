import comfy
import folder_paths
import torch
import os

from nodes import CheckpointLoaderSimple

from dfloat11 import DFloat11Model, compress_model

from .dfloat11_custom import DFloat11FluxDiffusersModel

from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux

from .pattern_dict import MODEL_TO_PATTERN_DICT

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
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)

        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_config = comfy.sd.model_detection.model_config_from_unet(state_dict, "")
        
        if model_config is None:
            # Assume it is CosmosPredict2, because no other model architectures are supported yet
            state_dict["blocks.0.mlp.layer1.weight"] = None
            model_config = comfy.sd.model_detection.model_config_from_unet(state_dict, "")
            assert model_config is not None, "Unable to detect model type"
        
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(state_dict, "")
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

class CheckpointLoaderWithDFloat11(CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The diffusion model in DF11 format."}),
            }
        }

    FUNCTION = "load_checkpoint_with_df11"
    CATEGORY = "DFloat11"
    DESCRIPTION = "Loads a diffusion model checkpoint, along with a DF11 unet."
    
    def load_checkpoint_with_df11(self, ckpt_name, dfloat11_model_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        diffusion_model, clip, vae, *_ = out

        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)
        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[type(diffusion_model.model).__name__],
            bfloat16_model=diffusion_model.model.diffusion_model,
            device=offload_device,
        )
        
        return (diffusion_model, clip, vae)



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


class DFloat11CheckpointCompressor:
    """
    A custom node to compress a DFloat11 diffusion model (unet only) from the `checkpoints` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the checkpoint (model) to load."}),
                "model_type": (list(MODEL_TO_PATTERN_DICT.keys()),)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_bfloat16_checkpoint"
    CATEGORY = "DFloat11"

    def load_bfloat16_checkpoint(self, ckpt_name, model_type):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        state_dict, metadata = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)

        diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
        parameters = comfy.utils.calculate_parameters(state_dict, diffusion_model_prefix)
        weight_dtype = comfy.utils.weight_dtype(state_dict, diffusion_model_prefix)
        
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_config = comfy.model_detection.model_config_from_unet(state_dict, diffusion_model_prefix, metadata=metadata)
        model_config.set_inference_dtype(weight_dtype, weight_dtype)
        
        unet_weight_dtype = list(model_config.supported_inference_dtypes)
        
        unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
        
        inital_load_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
        model = model_config.get_model(state_dict, diffusion_model_prefix, device=inital_load_device)
        
        model.load_model_weights(state_dict, diffusion_model_prefix)
        model_patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)

        diffusion_model = model_patcher.model.diffusion_model
        
        save_path = f"{os.path.splitext(ckpt_path)[0]}-DF11"

        compress_model(
            model=diffusion_model,
            pattern_dict=MODEL_TO_PATTERN_DICT[model_type],
            save_path=save_path,
            save_single_file=True,
            check_correctness=True,
            block_range=(0, 500),
        )

        return (save_path,)
