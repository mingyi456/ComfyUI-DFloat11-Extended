import comfy
import folder_paths
import torch

from dfloat11 import DFloat11Model, compress_model

from .dfloat11_custom import DFloat11FluxDiffusersModel

from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux

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
    "FluxSchnell": {
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
    "Chroma": {
        "double_blocks\.\d+": (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ),
        "single_blocks\.\d+": (
            "linear1",
            "linear2",
        ),
    },
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
        print("unet_config =", comfy.model_detection.detect_unet_config(sd, ""), sep = "\n")
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(sd, "")
        model = model.to(offload_device)
        print(f"pattern_dict type = {type(model_config).__name__}")

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

        # model_config = comfy.sd.model_detection.model_config_from_unet(state_dict, "")
        
                
        unet_config = {'image_model': 'flux', 'in_channels': 16, 'patch_size': 2, 'out_channels': 16, 'vec_in_dim': 768, 'context_in_dim': 4096, 'hidden_size': 3072, 'mlp_ratio': 4.0, 'num_heads': 24, 'depth': 19, 'depth_single_blocks': 38, 'axes_dim': [16, 56, 56], 'theta': 10000, 'qkv_bias': True, 'guidance_embed': True}
        unet_config["guidance_embed"] = "time_text_embed.guidance_embedder.linear_1.weight" in state_dict
        
        model_config = comfy.supported_models.Flux(unet_config)
        
        
        model_config.set_inference_dtype(torch.bfloat16, torch.bfloat16)
        model = model_config.get_model(state_dict, "")
        model = model.to(offload_device)
        print(f"pattern_dict type = {type(model_config).__name__}")

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
                "model_type": (["Flux", "Chroma"],)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_bfloat16_model"
    CATEGORY = "DFloat11"

    def load_bfloat16_model(self, bfloat16_model_name, model_type):
        bfloat16_model_path = folder_paths.get_full_path_or_raise("diffusion_models", bfloat16_model_name)
        # sd = comfy.utils.load_torch_file(bfloat16_model_path)
        

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        
        print("Loading model")

        model = comfy.sd.load_diffusion_model(bfloat16_model_path, {"dtype" : torch.bfloat16})
        # model = model.to(offload_device)

        save_path = f"{bfloat16_model_path}-DF11"
        
        compress_model(
            model=model.model.diffusion_model,
            pattern_dict=MODEL_TO_PATTERN_DICT[model_type],
            save_path= save_path,
            save_single_file= True,
            check_correctness= True,
            block_range= (0, 100),
        )

        return (save_path,)

'''
class UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "DFloat11"

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
'''

"""
dir(model) = ['__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_load_list', 'add_callback', 'add_callback_with_key', 'add_hook_patches', 'add_object_patch', 'add_patches', 'add_weight_wrapper', 'add_wrapper', 'add_wrapper_with_key', 'additional_models', 'apply_hooks', 'attachments', 'backup', 'cached_hook_patches', 'calculate_weight', 'callbacks', 'clean_hooks', 'cleanup', 'clear_cached_hook_weights', 'clone', 'clone_has_same_weights', 'current_hooks', 'current_loaded_device', 'detach', 'eject_model', 'force_cast_weights', 'forced_hooks', 'get_additional_models', 'get_additional_models_with_key', 'get_all_callbacks', 'get_all_wrappers', 'get_attachment', 'get_callbacks', 'get_combined_hook_patches', 'get_injections', 'get_key_patches', 'get_model_object', 'get_nested_additional_models', 'get_wrappers', 'hook_backup', 'hook_mode', 'hook_patches', 'hook_patches_backup', 'inject_model', 'injections', 'is_clip', 'is_clone', 'is_injected', 'load', 'load_device', 'loaded_size', 'lowvram_patch_counter', 'memory_required', 'model', 'model_dtype', 'model_options', 'model_patches_models', 'model_patches_to', 'model_size', 'model_state_dict', 'object_patches', 'object_patches_backup', 'offload_device', 'parent', 'partially_load', 'partially_unload', 'patch_cached_hook_weights', 'patch_hook_weight_to_device', 'patch_hooks', 'patch_model', 'patch_weight_to_device', 'patches', 'patches_uuid', 'pre_run', 'prepare_hook_patches_current_keyframe', 'prepare_state', 'register_all_hook_patches', 'remove_additional_models', 'remove_attachments', 'remove_callbacks_with_key', 'remove_injections', 'remove_wrappers_with_key', 'restore_hook_patches', 'set_additional_models', 'set_attachments', 'set_hook_mode', 'set_injections', 'set_model_attn1_output_patch', 'set_model_attn1_patch', 'set_model_attn1_replace', 'set_model_attn2_output_patch', 'set_model_attn2_patch', 'set_model_attn2_replace', 'set_model_compute_dtype', 'set_model_denoise_mask_function', 'set_model_double_block_patch', 'set_model_emb_patch', 'set_model_forward_timestep_embed_patch', 'set_model_input_block_patch', 'set_model_input_block_patch_after_skip', 'set_model_output_block_patch', 'set_model_patch', 'set_model_patch_replace', 'set_model_post_input_patch', 'set_model_sampler_calc_cond_batch_function', 'set_model_sampler_cfg_function', 'set_model_sampler_post_cfg_function', 'set_model_sampler_pre_cfg_function', 'set_model_unet_function_wrapper', 'size', 'skip_injection', 'unpatch_hooks', 'unpatch_model', 'use_ejected', 'weight_inplace_update', 'weight_wrapper_patches', 'wrappers']

dir(model.model) = ['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_apply_model', '_backward_hooks', '_backward_pre_hooks', '_buffers', '_call_impl', '_compiled_call_impl', '_forward_hooks', '_forward_hooks_always_called', '_forward_hooks_with_kwargs', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', '_is_full_backward_hook', '_load_from_state_dict', '_load_state_dict_post_hooks', '_load_state_dict_pre_hooks', '_maybe_warn_non_full_backward_hook', '_modules', '_named_members', '_non_persistent_buffers_set', '_parameters', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_state_dict_hooks', '_state_dict_pre_hooks', '_version', '_wrapped_call_impl', 'add_module', 'adm_channels', 'apply', 'apply_model', 'bfloat16', 'buffers', 'call_super_init', 'children', 'compile', 'concat_cond', 'concat_keys', 'cpu', 'cuda', 'current_patcher', 'current_weight_patches_uuid', 'device', 'diffusion_model', 'double', 'dump_patches', 'encode_adm', 'eval', 'extra_conds', 'extra_conds_shapes', 'extra_repr', 'float', 'forward', 'get_buffer', 'get_dtype', 'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu', 'latent_format', 'load_model_weights', 'load_state_dict', 'lowvram_patch_counter', 'manual_cast_dtype', 'memory_required', 'memory_usage_factor', 'memory_usage_factor_conds', 'memory_usage_shape_process', 'model_config', 'model_loaded_weight_memory', 'model_lowvram', 'model_sampling', 'model_type', 'modules', 'mtia', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'process_latent_in', 'process_latent_out', 'process_timestep', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 'register_load_state_dict_pre_hook', 'register_module', 'register_parameter', 'register_state_dict_post_hook', 'register_state_dict_pre_hook', 'requires_grad_', 'scale_latent_inpaint', 'set_extra_state', 'set_inpaint', 'set_submodule', 'share_memory', 'state_dict', 'state_dict_for_saving', 'to', 'to_empty', 'train', 'training', 'type', 'xpu', 'zero_grad']
"""


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
