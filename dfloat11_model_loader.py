import comfy
import folder_paths
import torch
import os

from nodes import CheckpointLoaderSimple, UNETLoader
from dfloat11 import DFloat11Model, compress_model
from .dfloat11_custom import DFloat11ModelPatcher
from .dfloat11_diffusers import DFloat11FluxDiffusersModel
from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux
from .pattern_dict import MODEL_TO_PATTERN_DICT

class DFloat11ModelLoaderAdvanced:
    """
    A custom node to load a DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

    '''
    max_memory: Maximum memory allocation per device
    cpu_offload: Enables CPU offloading; only keeps a single block of weights in GPU at once
    cpu_offload_blocks: Number of transformer blocks to offload to CPU; if None, offload all blocks
    pin_memory: Enables memory-pinning/page-locking when using CPU offloading
    '''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to offload to CPU RAM"}),
                "cpu_offload_blocks": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "If set to 0, all blocks will be offloaded to CPU RAM"}),
                "pin_memory": ("BOOLEAN", {"default": True, "tooltip": "Whether to lock/pin the weights to CPU RAM. Enabling this option increases RAM usage (which might cause OOM), but should increase speed"}),
                "custom_modelpatcher": ("BOOLEAN", {"default": True, "tooltip": "Whether to use the experimental custom ModelPatcher. Currently has no effect since disabling it will cause errors"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_dfloat11_model_advanced"
    CATEGORY = "DFloat11"

    def load_dfloat11_model_advanced(self, dfloat11_model_name, cpu_offload, cpu_offload_blocks, pin_memory, custom_modelpatcher):
        if not cpu_offload:
            cpu_offload_blocks = 0
            pin_memory = True
        
        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)

        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        model_config = comfy.sd.model_detection.model_config_from_unet(state_dict, "")
        
        if model_config is None:
            # Assume it is CosmosPredict2, because no other possible model architectures are supported yet
            state_dict["blocks.0.mlp.layer1.weight"] = None
            model_config = comfy.sd.model_detection.model_config_from_unet(state_dict, "")
            assert model_config is not None, "Unable to detect model type"
        
        df11_type = type(model_config).__name__
        
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(state_dict, "")
        model = model.to(offload_device)

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[df11_type],
            bfloat16_model=model.diffusion_model,
            device=offload_device,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks if cpu_offload_blocks > 0 else None,
            pin_memory=pin_memory,
        )

        # Always use DFloat11ModelPatcher for DF11 models (required due to missing .weight attributes

        return (
            DFloat11ModelPatcher(model, load_device=load_device, offload_device=offload_device),
        )

class DFloat11ModelLoader(DFloat11ModelLoaderAdvanced):
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

    FUNCTION = "load_dfloat11_model"
    
    def load_dfloat11_model(self, dfloat11_model_name):
        return self.load_dfloat11_model_advanced(dfloat11_model_name, cpu_offload = False, cpu_offload_blocks = 0, pin_memory = True, custom_modelpatcher = True)


class DFloat11WanModelLoader:
    """
    A custom node to load a DFloat11 diffusion model from the `diffusion_models` directory.

    DFloat11 models are >30% smaller than their float16 counterparts, yet produce bit-for-bit identical outputs.
    """

    '''
    max_memory: Maximum memory allocation per device
    cpu_offload: Enables CPU offloading; only keeps a single block of weights in GPU at once
    cpu_offload_blocks: Number of transformer blocks to offload to CPU; if None, offload all blocks
    pin_memory: Enables memory-pinning/page-locking when using CPU offloading
    '''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to offload to CPU RAM"}),
                "cpu_offload_blocks": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "If set to 0, all blocks will be offloaded to CPU RAM"}),
                "pin_memory": ("BOOLEAN", {"default": True, "tooltip": "Whether to lock/pin the weights to CPU RAM. Enabling this option increases RAM usage (which might cause OOM), but should increase speed"}),
                "custom_modelpatcher": ("BOOLEAN", {"default": True, "tooltip": "Whether to use the experimental custom ModelPatcher. Currently has no effect since disabling it will cause errors"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_dfloat11_model_advanced"
    CATEGORY = "DFloat11"

    def load_dfloat11_model_advanced(self, dfloat11_model_name, cpu_offload, cpu_offload_blocks, pin_memory, custom_modelpatcher):
        if not cpu_offload:
            cpu_offload_blocks = 0
            pin_memory = True
        
        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)

        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        missing = {"blocks.0.ffn.0.weight" : torch.empty(state_dict['blocks.0.ffn.0.bias'].shape[0], 1, device = "meta")}

        model_config = comfy.sd.model_detection.model_config_from_unet(state_dict | missing, "")
        
        df11_type = type(model_config).__name__
        
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(state_dict, "")
        model = model.to(offload_device)

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[df11_type],
            bfloat16_model=model.diffusion_model,
            device=offload_device,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks if cpu_offload_blocks > 0 else None,
            pin_memory=pin_memory,
        )

        # Always use DFloat11ModelPatcher for DF11 models (required due to missing .weight attributes

        return (
            DFloat11ModelPatcher(model, load_device=load_device, offload_device=offload_device),
        )



def load_diffusion_model_state_dict(sd, model_options={}, metadata=None):
    """
    Loads a UNet diffusion model from a state dictionary, supporting both diffusers and regular formats.

    Args:
        sd (dict): State dictionary containing model weights and configuration
        model_options (dict, optional): Additional options for model loading. Supports:
            - dtype: Override model data type
            - custom_operations: Custom model operations
            - fp8_optimizations: Enable FP8 optimizations

    Returns:
        ModelPatcher: A wrapped model instance that handles device management and weight loading.
        Returns None if the model configuration cannot be detected.

    The function:
    1. Detects and handles different model formats (regular, diffusers, mmdit)
    2. Configures model dtype based on parameters and device capabilities
    3. Handles weight conversion and device placement
    4. Manages model optimization settings
    5. Loads weights and returns a device-managed model instance
    """
    dtype = model_options.get("dtype", None)

    #Allow loading unets from checkpoint files
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)

    load_device = comfy.model_management.get_torch_device()
    model_config = comfy.model_detection.model_config_from_unet(sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = sd
    else:
        raise Exception()
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None: #diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else: #diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = comfy.model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(None, load_device, model_config.supported_inference_dtypes)
    else:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    print(f"model_config.set_inference_dtype({unet_dtype}, {manual_cast_dtype})")
    model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)
    model.load_model_weights(new_sd, "")
    left_over = sd.keys()
    if len(left_over) > 0:
        logging.info("left over keys in diffusion model: {}".format(left_over))
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


class UNETLoaderWithDFloat11(UNETLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                    "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                    "weight_dtype": (["default", "bf16"],),
                    "use_df11": ("BOOLEAN", {"default": True, "tooltip": "Whether to use DF11"}),
                    "wipe_bf16_sd": ("BOOLEAN", {"default": False, "tooltip": "Whether to zero out bf16 state_dict"}),
                    "dfloat11_model_name": (folder_paths.get_filename_list("diffusion_models"), ),
                    "cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to offload to CPU RAM"}),
                    "cpu_offload_blocks": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1, "tooltip": "If set to 0, all blocks will be offloaded to CPU RAM"}),
                    "pin_memory": ("BOOLEAN", {"default": True, "tooltip": "Whether to lock/pin the weights to CPU RAM. Enabling this option increases RAM usage (which might cause OOM), but should increase speed"}),
                    }
                }
    FUNCTION = "load_unet_with_df11"

    CATEGORY = "DFloat11"

    def load_unet_with_df11(self, unet_name, weight_dtype, use_df11, wipe_bf16_sd, dfloat11_model_name, cpu_offload, cpu_offload_blocks, pin_memory):
        model_options = {}
        if weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        # model_patcher = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        
        sd, metadata = comfy.utils.load_torch_file(unet_path, return_metadata=True)
        
        if use_df11 and wipe_bf16_sd:
            for tensor in sd.values():
                tensor.zero_()
        
        model_patcher = load_diffusion_model_state_dict(sd, model_options=model_options, metadata=metadata)
        if model_patcher is None:
            logging.error("ERROR UNSUPPORTED DIFFUSION MODEL {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}\n{}".format(unet_path, model_detection_error_hint(unet_path, sd)))
        
        if not use_df11:
            return (model_patcher,)

        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)
        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        df11_type = type(model_patcher.model.model_config).__name__
        
        df11_model_patcher = DFloat11ModelPatcher(
            model_patcher.model,
            load_device=load_device,
            offload_device=offload_device,
        )

        del model_patcher

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[df11_type],
            bfloat16_model=df11_model_patcher.model.diffusion_model,
            device=offload_device,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks if cpu_offload_blocks > 0 else None,
            pin_memory=pin_memory,
        )

        return (df11_model_patcher,)

"""
class UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
"""

class DFloat11LoadingPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_patcher": ("MODEL", {"tooltip": "The model to display information for"}),
                "load_version": (["v1", "v1.5", "v2"],),
                "memory_usage_factor_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step":0.01, "round": 0.01, "tooltip": "The multiplier to scale ComfyUI's memory usage estimation"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_loading_methods"
    CATEGORY = "DFloat11"

    def patch_loading_methods(self, model_patcher, load_version, memory_usage_factor_scale):
        
        new_model_patcher = model_patcher.clone()
        new_model_patcher.patch_loading_methods(load_version)
        
        new_model_patcher.model.model_config.memory_usage_factor *= memory_usage_factor_scale
        new_model_patcher.model.memory_usage_factor = new_model_patcher.model.model_config.memory_usage_factor
        
        return (new_model_patcher,)

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
        model_patcher, clip, vae, *_ = out

        dfloat11_model_path = folder_paths.get_full_path_or_raise("diffusion_models", dfloat11_model_name)
        state_dict = comfy.utils.load_torch_file(dfloat11_model_path)
        if not any(k.endswith("encoded_exponent") for k in state_dict.keys()):
            raise ValueError(f"The model '{dfloat11_model_name}' is not a DFloat11 model.")

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        df11_type = type(model_patcher.model).__name__
        
        df11_model_patcher = DFloat11ModelPatcher(
            model_patcher.model,
            load_device=load_device,
            offload_device=offload_device,
        )
        
        del model_patcher

        DFloat11Model.from_single_file(
            dfloat11_model_path,
            pattern_dict=MODEL_TO_PATTERN_DICT[df11_type],
            bfloat16_model=df11_model_patcher.model.diffusion_model,
            device=offload_device,
        )
        
        return (df11_model_patcher, clip, vae)



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
            'axes_dim': [16, 56, 56], 
            'num_heads': 24, 
            'mlp_ratio': 4.0, 
            'theta': 10000, 
            'out_channels': 16, 
            'qkv_bias': True, 
            'txt_ids_dims': [], 
            'in_channels': 16, 
            'hidden_size': 3072, 
            'context_in_dim': 4096, 
            'patch_size': 2, 
            'vec_in_dim': 768, 
            'depth': 19, 
            'depth_single_blocks': 38, 
            'guidance_embed': True, 
            'yak_mlp': False, 
            'txt_norm': False
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
            DFloat11ModelPatcher(model, load_device=load_device, offload_device=offload_device),
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
