import comfy
import comfy.lora
import comfy.float
import folder_paths
import torch
import torch.nn as nn

import os

import cupy as cp
import pkg_resources
import re
import math
import uuid

from sys import stderr

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory

from safetensors.torch import load_file, save_file

from tqdm import tqdm

from typing import Optional, Dict, Union, List, Tuple, Any

from dfloat11 import DFloat11Model

from dfloat11.dfloat11 import TensorManager, get_no_split_classes

from dfloat11.dfloat11 import version, threads_per_block, bytes_per_thread

from dfloat11.dfloat11_utils import get_codec, get_32bit_codec, get_luts, encode_weights

from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux

ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')


# Store for LoRA patches that need to be applied after decompression
# Key: module id -> List of (weight_key, patch_list) tuples
_module_lora_patches: Dict[int, List[Tuple[str, List, Any]]] = {}


def register_lora_for_df11_module(module: nn.Module, weight_key: str, patch_list: List, submodule: nn.Module = None):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    Registers LoRA patches to be applied after DFloat11 decompression.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    """
    module_id = id(module)
    if module_id not in _module_lora_patches:
        _module_lora_patches[module_id] = []
    
    _module_lora_patches[module_id].append((weight_key, patch_list, submodule or module))


def clear_lora_for_df11_module(module: nn.Module):
    """Clear all registered LoRA patches for a module."""
    module_id = id(module)
    if module_id in _module_lora_patches:
        del _module_lora_patches[module_id]


def clear_all_df11_lora_patches():
    """Clear all registered LoRA patches."""
    _module_lora_patches.clear()


def string_to_seed(s: str) -> int:
    """Convert a string to a deterministic seed value."""
    import hashlib
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)


def apply_lora_to_weight(weight: torch.Tensor, weight_key: str, patch_list: List) -> torch.Tensor:
    """
    Apply LoRA patches to a weight tensor.
    
    Args:
        weight: The decompressed weight tensor
        weight_key: The key for this weight
        patch_list: List of LoRA patches to apply
        
    Returns:
        The patched weight tensor
    """
    if not patch_list:
        return weight
    
    # Use ComfyUI's calculate_weight function
    patched_weight = comfy.lora.calculate_weight(patch_list, weight, weight_key)
    
    # Apply stochastic rounding to maintain dtype
    result = comfy.float.stochastic_rounding(patched_weight, weight.dtype, seed=string_to_seed(weight_key))
    
    return result


def get_hook_flux_diffusers(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly
    and optionally applies LoRA patches after decompression.
    
    Args:
        threads_per_block: CUDA thread configuration 
        bytes_per_thread: Number of bytes processed per CUDA thread
        
    Returns:
        A forward pre-hook function for PyTorch modules
    """
    threads_per_block = tuple(threads_per_block)

    def decode_hook(module, _):
        device = module.luts.device

        # Load offloaded tensors to GPU if not already there
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name, tensor in module.offloaded_tensors.items():
                if not (
                    hasattr(module, tensor_name) and (getattr(module, tensor_name).device == device)
                ):
                    module.register_buffer(tensor_name, tensor.to(device, non_blocking=True))

        # Get dimensions for tensor reconstruction
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]

        # Get output tensor for reconstructed weights
        reconstructed = TensorManager.allocate_bfloat16(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        # Launch CUDA kernel to decode the compressed weights
        with cp.cuda.Device(device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=module.shared_mem_size, args=[
                module.luts.data_ptr(),
                module.encoded_exponent.data_ptr(),
                module.sign_mantissa.data_ptr(),
                module.output_positions.data_ptr(),
                module.gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])

        # Get LoRA patches for this module (if any)
        module_id = id(module)
        lora_patches = _module_lora_patches.get(module_id, [])
        
        # Build a lookup dict for faster LoRA matching
        # Maps submodule -> (weight_key, patch_list)
        lora_by_submodule = {}
        for weight_key, patch_list, submodule in lora_patches:
            submodule_id = id(submodule)
            if submodule_id not in lora_by_submodule:
                lora_by_submodule[submodule_id] = []
            lora_by_submodule[submodule_id].append((weight_key, patch_list))

        # Inject reconstructed weights into the appropriate module
        # Handle special case where weights need to be split across multiple submodules
        
        if len(module.weight_injection_modules) == 10:
            # Should be double_block
            
            reconstructed[:] = torch.cat((
                reconstructed[0:141557760], 
                reconstructed[160432128:169869312], 
                reconstructed[141557760:160432128], 
                reconstructed[169869312:]
            ))
            
            weights = torch.tensor_split(reconstructed, module.split_positions)
            
            # Define the mapping: (index, weight_slice_or_custom, out_features, in_features)
            weight_assignments = [
                (0, weights[0], None, None),  # img_mod.lin
                (1, reconstructed[113246208:141557760], 9216, 3072),  # img_attn.qkv
                (2, weights[8], None, None),  # img_attn.proj
                (3, weights[10], None, None),  # img_mlp.0
                (4, weights[11], None, None),  # img_mlp.2
                (5, weights[1], None, None),  # txt_mod.lin
                (6, reconstructed[141557760:169869312], 9216, 3072),  # txt_attn.qkv
                (7, weights[9], None, None),  # txt_attn.proj
                (8, weights[12], None, None),  # txt_mlp.0
                (9, weights[13], None, None),  # txt_mlp.2
            ]
            
            for idx, weight_data, custom_out, custom_in in weight_assignments:
                submodule = module.weight_injection_modules[idx]
                
                if custom_out is not None:
                    final_weight = weight_data.view(custom_out, custom_in)
                else:
                    final_weight = weight_data.view(submodule.out_features, submodule.in_features)
                
                # Apply LoRA if registered for this submodule
                submodule_id = id(submodule)
                if submodule_id in lora_by_submodule:
                    for weight_key, patch_list in lora_by_submodule[submodule_id]:
                        final_weight = apply_lora_to_weight(final_weight, weight_key, patch_list)
                
                submodule.weight = final_weight

        elif len(module.weight_injection_modules) == 3:
            # Should be single_block
            
            reconstructed[:] = torch.cat((
                reconstructed[113246208:141557760], 
                reconstructed[28311552:66060288], 
                reconstructed[66060288:113246208], 
                reconstructed[0:28311552]
            ))
            
            weights = torch.tensor_split(reconstructed, module.split_positions)
            
            # Define assignments for single_block
            weight_assignments = [
                (0, reconstructed[0:66060288], 21504, 3072),  # linear1
                (1, weights[2], None, None),  # linear2
                (2, reconstructed[113246208:141557760], 9216, 3072),  # modulation.lin
            ]
            
            for idx, weight_data, custom_out, custom_in in weight_assignments:
                submodule = module.weight_injection_modules[idx]
                
                if custom_out is not None:
                    final_weight = weight_data.view(custom_out, custom_in)
                else:
                    final_weight = weight_data.view(submodule.out_features, submodule.in_features)
                
                # Apply LoRA if registered for this submodule
                submodule_id = id(submodule)
                if submodule_id in lora_by_submodule:
                    for weight_key, patch_list in lora_by_submodule[submodule_id]:
                        final_weight = apply_lora_to_weight(final_weight, weight_key, patch_list)
                
                submodule.weight = final_weight
        
        else:
            raise Exception(f"{len(module.weight_injection_modules)} weight_injection_modules \n{module.weight_injection_modules}")

        # Delete tensors from GPU if offloading is enabled
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name in module.offloaded_tensors.keys():
                if hasattr(module, tensor_name):
                    tmp = getattr(module, tensor_name)
                    delattr(module, tensor_name)
                    del tmp

    return decode_hook


def load_and_replace_tensors_flux_diffusers(
    model,
    directory_path,
    dfloat11_config,
    cpu_offload=False,
    cpu_offload_blocks=None,
    pin_memory=True,
    from_single_file=False,
):
    """
    Loads DFloat11 compressed weights from safetensors files and configures the model
    to use them with on-the-fly decompression.
    
    Args:
        model: The PyTorch model to load weights into
        directory_path: Path to the directory containing safetensors files
        dfloat11_config: Configuration for DFloat11 compression
        
    Returns:
        The model with configured DFloat11 compression
    """
    threads_per_block = dfloat11_config['threads_per_block']
    bytes_per_thread  = dfloat11_config['bytes_per_thread']
    pattern_dict      = dfloat11_config['pattern_dict']
    
    # Get all .safetensors files in the directory
    safetensors_files = [
        f for f in os.listdir(directory_path) if f.endswith('.safetensors')
    ] if not from_single_file else [directory_path]
    loading_desc = 'Loading DFloat11 safetensors'
    if cpu_offload:
        loading_desc += ' (offloaded to CPU'
        if pin_memory:
            loading_desc += ', memory pinned'
        loading_desc += ')'

    for file_name in tqdm(safetensors_files, desc=loading_desc):
        file_path = os.path.join(directory_path, file_name) if not from_single_file else file_name
        
        # Load the tensors from the file
        loaded_tensors = convert_diffusers_to_comfyui_flux(load_file(file_path))
        
        # Iterate over each tensor in the file
        for tensor_name, tensor_value in loaded_tensors.items():
            # Check if this tensor exists in the model's state dict
            if tensor_name in model.state_dict():
                # Get the parameter or buffer
                if tensor_name in dict(model.named_parameters()):
                    # It's a parameter, we can set it directly
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}", file=stderr)
                else:
                    # It's a buffer, we can also set it directly
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}", file=stderr)
            else:
                # Split the tensor name to get module path
                parts = tensor_name.split('.')
                module = model
                
                # Navigate to the correct module
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        print(f"Cannot find module path for {tensor_name}", file=stderr)
                        break
                else:
                    if parts[-1] == 'split_positions':
                        setattr(module, 'split_positions', tensor_value.tolist())
                    else:
                        if cpu_offload and (cpu_offload_blocks is None or cpu_offload_blocks > 0) and parts[-1] in offloaded_tensor_names:
                            if not hasattr(module, 'offloaded_tensors'):
                                setattr(module, 'offloaded_tensors', {})

                            module.offloaded_tensors[parts[-1]] = tensor_value.pin_memory() if pin_memory else tensor_value

                            if (cpu_offload_blocks is not None) and (cpu_offload_blocks > 0) and (len(module.offloaded_tensors) == len(offloaded_tensor_names)):
                                cpu_offload_blocks -= 1
                        else:
                            # Register the buffer to the found module
                            module.register_buffer(parts[-1], tensor_value)

                    # Set up decompression for encoded weights
                    if parts[-1] == 'encoded_exponent':
                        # Register the decode hook to decompress weights during forward pass
                        module.register_forward_pre_hook(get_hook_flux_diffusers(threads_per_block, bytes_per_thread))

                        # Configure weight injection based on module type
                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, '.'.join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    # Remove weight attribute from embedding layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    # Remove weight attribute from linear layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                else:
                                    # Handle special case for multi-module weight injection
                                    setattr(module, 'weight_injection_modules', [])
                                    for attr_path in attr_names:
                                        parts = attr_path.split('.')
                                        target = module
                                        for p in parts:
                                            target = getattr(target, p)

                                        tmp = target.weight
                                        delattr(target, 'weight')
                                        del tmp
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == 'output_positions':
                        # Calculate required shared memory size for CUDA kernel
                        output_positions_np = tensor_value.view(torch.uint32).numpy()
                        setattr(
                            module,
                            'shared_mem_size',
                            threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
                        )
    
    return model


class DFloat11FluxDiffusersModel(DFloat11Model):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def from_pretrained(
        cls,
        dfloat11_model_name_or_path: str,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        bfloat16_model = None,
        cpu_offload: bool = False,
        cpu_offload_blocks: Optional[int] = None,
        pin_memory: bool = True,
        from_single_file: bool = False,
        pattern_dict: Optional[dict[str, list[str]]] = None,
        **kwargs,
    ):
        """
        Load a model with DFloat11 compressed weights from local path or Hugging Face Hub.
        
        Args:
            dfloat11_model_name_or_path: Local path or HF Hub model name
            device: Target device for the model
            device_map: Strategy for distributing model across devices
            max_memory: Maximum memory allocation per device
            bfloat16_model: Optional pre-initialized model to load weights into
            cpu_offload: Enables CPU offloading; only keeps a single block of weights in GPU at once
            cpu_offload_blocks: Number of transformer blocks to offload to CPU; if None, offload all blocks
            pin_memory: Enables memory-pinning/page-locking when using CPU offloading
            from_single_file: Whether to load a single safetensors file
            pattern_dict: Dictionary mapping regex patterns to submodule lists
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_config
            
        Returns:
            Model with DFloat11 compressed weights configured for on-the-fly decompression
        """
        # Resolve model path, downloading from HF Hub if needed
        if from_single_file:
            if os.path.isfile(dfloat11_model_name_or_path):
                dfloat11_model_path = dfloat11_model_name_or_path
            elif os.path.isdir(dfloat11_model_name_or_path):
                raise IsADirectoryError(f'Expected `dfloat11_model_name_or_path` to be the path to a safetensors file, but found a directory: "{dfloat11_model_name_or_path}".')
            else:
                raise FileNotFoundError(f'The file "{dfloat11_model_name_or_path}" does not exist.')
        else:
            if os.path.exists(dfloat11_model_name_or_path):
                dfloat11_model_path = dfloat11_model_name_or_path
            else:
                dfloat11_model_path = dfloat11_model_name_or_path.replace('/', '__')
                if not os.path.exists(dfloat11_model_path):
                    snapshot_download(dfloat11_model_name_or_path, local_dir=dfloat11_model_path)
                    
        print("Using overriden DFloat11FluxDiffusersModel class")

        # Load model configuration
        if bfloat16_model:
            if from_single_file:
                config = {
                    'dfloat11_config': {
                        'version': version,
                        'threads_per_block': threads_per_block,
                        'bytes_per_thread': bytes_per_thread,
                        'pattern_dict': pattern_dict,
                    },
                }
            else:
                with open(os.path.join(dfloat11_model_path, 'config.json'), 'r', encoding='utf-8') as f:
                    config = json.load(f)

            model = bfloat16_model
        else:
            raise Exception("`bfloat16_model` must be specified")

        # Verify model has DFloat11 configuration
        if isinstance(config, dict) and 'dfloat11_config' in config:
            dfloat11_config = config['dfloat11_config']
        elif hasattr(config, 'dfloat11_config'):
            dfloat11_config = config.dfloat11_config
        else:
            raise AttributeError('"dfloat11_config" not found: it is expected to be found in the config file or passed as an argument.')

        # Load compressed weights and configure decompression
        load_and_replace_tensors_flux_diffusers(
            model, dfloat11_model_path, dfloat11_config,
            cpu_offload=cpu_offload, cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory, from_single_file=from_single_file,
        )

        if not cpu_offload:
            # Calculate and report model size
            model_bytes = 0
            for param in model.state_dict().values():
                model_bytes += param.nbytes

            print(f"Total model size: {model_bytes / 1e9:0.4f} GB", file=stderr)

        # Move model to specified device or distribute across multiple devices
        if device:
            model = model.to(device)
        else:
            assert device_map == 'auto', "device_map should be 'auto' if no specific device is provided."
            # Identify modules that must remain on same device for decompression
            no_split_classes = get_no_split_classes(model, dfloat11_config['pattern_dict'])
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            model = dispatch_model(model, device_map)

            # Warn if model is not fully on GPU
            if any(param.device.type == 'cpu' for param in model.parameters()):
                print("Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.", file=stderr)

        return model


import logging
import inspect
from comfy.model_patcher import LowVramPatch, move_weight_functions, wipe_lowvram_weight, get_key_weight, string_to_seed
from comfy.patcher_extension import CallbacksMP


def df11_module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    if hasattr(module, "encoded_exponent"):
        module_mem += module.encoded_exponent.nelement()
        module_mem += module.sign_mantissa.nelement()
        
    return module_mem


class DFloat11ModelPatcher(comfy.model_patcher.ModelPatcher):
    """
    Base ModelPatcher for all DFloat11 compressed models.
    Handles the generic DFloat11 weight format that removes the 'weight' attribute
    from compressed layers and uses custom decompression hooks.
    
    This class MUST be used for all DFloat11 models because the standard ModelPatcher
    will fail when trying to access .weight on compressed layers.
    """
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size=size, weight_inplace_update=weight_inplace_update)

        self.model.state_dict = self._patch_state_dict(self.model.state_dict)
        
        # Build mapping from weight keys to their DF11 parent modules and submodules
        self._df11_weight_to_module_map = self._build_df11_weight_map()

    def _build_df11_weight_map(self) -> Dict[str, Tuple[nn.Module, nn.Module]]:
        """
        Build a mapping from weight keys to (df11_parent_module, target_submodule).
        
        This allows us to register LoRA patches with the correct DF11 module
        so they get applied after decompression.
        
        Returns:
            Dict mapping weight key -> (df11_module, submodule)
        """
        weight_map = {}
        
        for name, module in self.model.named_modules():
            # Check if this is a DF11 compressed module
            if hasattr(module, 'weight_injection_modules') and hasattr(module, 'encoded_exponent'):
                # This module has multiple submodules that receive weights
                for submodule in module.weight_injection_modules:
                    # Find the full path to this submodule
                    for sub_name, sub_mod in self.model.named_modules():
                        if sub_mod is submodule:
                            weight_key = f"{sub_name}.weight"
                            weight_map[weight_key] = (module, submodule)
                            break
            elif hasattr(module, 'encoded_exponent'):
                # Single compressed module (Linear or Embedding)
                weight_key = f"{name}.weight"
                weight_map[weight_key] = (module, module)
        
        return weight_map

    def partially_unload(self, offload_device, memory_to_free=0):
        """
        DFloat11 compressed modules don't have a standard '.weight' attribute - 
        it's replaced with compressed tensors (encoded_exponent, sign_mantissa, etc.).
        ComfyUI's partial unloading mechanism uses get_key_weight() which fails
        on these modules, causing type comparison errors.
        
        TODO: Implement proper partial unloading that understands DFloat11's
        compressed tensor structure.
        """
        return 0
    
    def unpatch_hooks(self):
        """Clear LoRA patches when unpatching."""
        super().unpatch_hooks()
        # Clear all registered LoRA patches for DF11 modules
        clear_all_df11_lora_patches()

    def _patch_state_dict(self, state_dict_func):
        lora_loading_functions = {"model_lora_keys_unet", "add_patches"}
        from . import state_dict_shapes
        all_keys = set()
        for name in state_dict_shapes.__all__:
            keys_dict = getattr(state_dict_shapes, name)
            all_keys.update(keys_dict.keys())
            
        fake_state_dict = {f"diffusion_model.{key}": None for key in all_keys}
        
        def new_state_dict_func():
            call_stack = inspect.stack()
            caller_function = call_stack[1].function
            del call_stack
            if caller_function in lora_loading_functions:
                return fake_state_dict
            return state_dict_func()
        return new_state_dict_func

    def _load_list(self):
        loading = []
        for n, module in self.model.named_modules():
            params = []
            skip = False
            for name, param in module.named_parameters(recurse=False):
                params.append(name)
            for name, param in module.named_parameters(recurse=True):
                if name not in params:
                    skip = True
                    break
            if not skip and (hasattr(module, "comfy_cast_weights") or len(params) > 0):
                loading.append((comfy.model_management.module_size(module), n, module, params))

        return loading

    def _register_df11_lora_patches(self):
        """
        Register all LoRA patches with their corresponding DF11 modules.
        This should be called after patches are set but before the model runs.
        """
        # Clear any existing registrations
        clear_all_df11_lora_patches()
        
        # Go through all patches and register them with DF11 modules
        for weight_key, patch_list in self.patches.items():
            if weight_key in self._df11_weight_to_module_map:
                df11_module, submodule = self._df11_weight_to_module_map[weight_key]
                register_lora_for_df11_module(df11_module, weight_key, patch_list, submodule)
                logging.debug(f"Registered LoRA for DF11: {weight_key}")

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            
            # Register LoRA patches with DF11 modules BEFORE loading
            self._register_df11_lora_patches()
            
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                lowvram_weight = False

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_weight = True
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):
                            continue
                        
                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    # For DF11 modules, we don't use LowVramPatch - LoRA is handled in decode hook
                    if weight_key in self.patches:
                        if weight_key not in self._df11_weight_to_module_map:
                            # Non-DF11 weight, use standard patching
                            if force_patch_weights:
                                self.patch_weight_to_device(weight_key)
                            else:
                                _, set_func, convert_func = get_key_weight(self.model, weight_key)
                                m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                                patch_counter += 1
                        # DF11 weights are handled by the decode hook
                        
                    if bias_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(bias_key)
                        else:
                            _, set_func, convert_func = get_key_weight(self.model, bias_key)
                            m.bias_function = [LowVramPatch(bias_key, self.patches, convert_func, set_func)]
                            patch_counter += 1

                    cast_weight = True
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or mem_counter + module_mem < lowvram_model_memory:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    param_key = "{}.{}".format(n, param)
                    # Skip DF11 weights - they're handled by decode hook
                    if param_key not in self._df11_weight_to_module_map:
                        self.patch_weight_to_device(param_key, device_to=device_to)

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if lowvram_counter > 0:
                logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)
