import comfy
import folder_paths
import torch
import torch.nn as nn
import logging
import inspect
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

from typing import Optional, Dict, Union

from dfloat11 import DFloat11Model

from dfloat11.dfloat11 import TensorManager, get_no_split_classes

from dfloat11.dfloat11 import version, threads_per_block, bytes_per_thread

from dfloat11.dfloat11_utils import get_codec, get_32bit_codec, get_luts, encode_weights

ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')

from comfy.model_patcher import LowVramPatch, move_weight_functions, wipe_lowvram_weight, get_key_weight, string_to_seed
from comfy.patcher_extension import CallbacksMP

def get_hook_flux_diffusers(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    
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

        # Inject reconstructed weights into the appropriate module
        # Handle special case where weights need to be split across multiple submodules
        
            
        if len(module.weight_injection_modules) == 10:
            # Should be double_block
            
            reconstructed[:] = torch.cat((reconstructed[0:141557760], reconstructed[160432128:169869312], reconstructed[141557760:160432128], reconstructed[169869312:]))
            
            weights = torch.tensor_split(reconstructed, module.split_positions)
            
            module.weight_injection_modules[0].weight = weights[0].view(module.weight_injection_modules[0].out_features, module.weight_injection_modules[0].in_features) # img_mod.lin
            module.weight_injection_modules[1].weight = reconstructed[113246208 : 141557760].view(9216, 3072) # img_attn.qkv
            module.weight_injection_modules[2].weight = weights[8].view(module.weight_injection_modules[2].out_features, module.weight_injection_modules[2].in_features) # img_attn.proj
            module.weight_injection_modules[3].weight = weights[10].view(module.weight_injection_modules[3].out_features, module.weight_injection_modules[3].in_features) # img_mlp.0
            module.weight_injection_modules[4].weight = weights[11].view(module.weight_injection_modules[4].out_features, module.weight_injection_modules[4].in_features) # img_mlp.2
            module.weight_injection_modules[5].weight = weights[1].view(module.weight_injection_modules[5].out_features, module.weight_injection_modules[5].in_features) # txt_mod.lin
            module.weight_injection_modules[6].weight = reconstructed[141557760 : 169869312].view(9216, 3072) # txt_attn.qkv
            module.weight_injection_modules[7].weight = weights[9].view(module.weight_injection_modules[7].out_features, module.weight_injection_modules[7].in_features) # txt_attn.proj
            module.weight_injection_modules[8].weight = weights[12].view(module.weight_injection_modules[8].out_features, module.weight_injection_modules[8].in_features) # txt_mlp.0
            module.weight_injection_modules[9].weight = weights[13].view(module.weight_injection_modules[9].out_features, module.weight_injection_modules[9].in_features) # txt_mlp.2
            

        elif len(module.weight_injection_modules) == 3:
            # Should be single_block
            
            reconstructed[:] = torch.cat((reconstructed[113246208:141557760], reconstructed[28311552:66060288], reconstructed[66060288:113246208], reconstructed[0:28311552]))
            
            weights = torch.tensor_split(reconstructed, module.split_positions)
            module.weight_injection_modules[0].weight = reconstructed[0:66060288].view(21504, 3072) # linear1
            module.weight_injection_modules[1].weight = weights[2].view(module.weight_injection_modules[1].out_features, module.weight_injection_modules[1].in_features) # linear2
            module.weight_injection_modules[2].weight = reconstructed[113246208:141557760].view(9216, 3072) # modulation.lin
        
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
                                        # print(parts)
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

class CastBufferManager:
    """Manages a reusable float16 buffer for dtype conversion - mirrors TensorManager pattern"""
    _tensors = {}

    @staticmethod
    def get_float16_buffer(device, n_elements):
        if isinstance(device, str):
            device = torch.device(device)
        
        if device in CastBufferManager._tensors:
            existing = CastBufferManager._tensors[device]
            if existing.numel() >= n_elements:
                return existing[:n_elements]
            del CastBufferManager._tensors[device]
            torch.cuda.empty_cache()
        
        new_tensor = torch.empty(n_elements, dtype=torch.float16, device=device)
        CastBufferManager._tensors[device] = new_tensor
        return new_tensor


def get_hook_lora(patch_list, key):
    print(f"[HOOK SETUP] Creating LoRA hook for key: {key} with {len(patch_list)} patches")
    
    def lora_hook(module, input):
        weight = module.weight  # bfloat16 view into TensorManager buffer
        original_shape = weight.shape
        n_elements = weight.numel()
        
        # Get reusable fp16 buffer
        fp16_buffer = CastBufferManager.get_float16_buffer(weight.device, n_elements)
        fp16_buffer.copy_(weight.view(-1))
        temp_weight = fp16_buffer.view(original_shape)
        
        # Calculate LoRA - this creates a new tensor (unavoidable)
        try:
            new_weight = comfy.lora.calculate_weight(patch_list, temp_weight, key)
        except Exception as e:
            print(f"[LORA HOOK ERROR] Failed to calculate weight for {key}: {e}")
            raise e
        
        # Copy directly back into the ORIGINAL DFloat11 buffer (in-place)
        weight.view(-1).copy_(new_weight.view(-1).to(torch.bfloat16))
        
        del new_weight  # Free immediately
            
    return lora_hook


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
        # List to keep track of PyTorch hooks so we can remove them later
        self.lora_hook_handles = []
        self._last_patch_keys = None

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
        print(f"[DF11 UNPATCH] unpatch_hooks called. Keeping {len(self.lora_hook_handles)} LoRA handles")
        super().unpatch_hooks()

    def _check_weights_dirty(self):
        """Check if any module has weights modified by LoRA"""
        for n, m in self.model.named_modules():
            if getattr(m, '_df11_lora_modified', False):
                print(f"[DF11 CHECK] Found dirty module: {n}")
                return True
        return False

    def _mark_weights_dirty(self):
        """Mark modules as having LoRA-modified weights"""
        count = 0
        for n, m in self.model.named_modules():
            if hasattr(m, 'weight'):
                m._df11_lora_modified = True
                count += 1
        print(f"[DF11] Marked {count} modules as LoRA-modified")

    def _clear_lora_hooks(self):
        """Remove all LoRA hooks from modules"""
        # First, remove hooks we have handles for
        if hasattr(self, "lora_hook_handles"):
            for hook in self.lora_hook_handles:
                hook.remove()
            self.lora_hook_handles.clear()
        
        # CRITICAL: Also scan all modules and remove any lingering lora_hook functions
        # This catches hooks that persist when patcher instance is recreated
        removed_count = 0
        for n, m in self.model.named_modules():
            if hasattr(m, '_forward_pre_hooks'):
                hooks_to_remove = []
                for hook_id, hook_fn in m._forward_pre_hooks.items():
                    # Check if this is a lora_hook by checking the function name
                    if hasattr(hook_fn, '__name__') and 'lora_hook' in hook_fn.__name__:
                        hooks_to_remove.append(hook_id)
                    elif hasattr(hook_fn, '__qualname__') and 'lora_hook' in hook_fn.__qualname__:
                        hooks_to_remove.append(hook_id)
                
                for hook_id in hooks_to_remove:
                    del m._forward_pre_hooks[hook_id]
                    removed_count += 1
        
        print(f"[DF11] LoRA hooks cleared (handles: {len(self.lora_hook_handles) if hasattr(self, 'lora_hook_handles') else 0}, removed from modules: {removed_count})")

    def _reset_weights_to_base(self, device_to=None):
        """Force re-decode from DFloat11 by clearing TensorManager cache"""
        reset_count = 0
        dirty_found = 0
        
        # Count dirty modules
        dirty_modules = []
        for n, m in self.model.named_modules():
            if getattr(m, '_df11_lora_modified', False):
                dirty_modules.append((n, m))
                dirty_found += 1
        
        if dirty_found == 0:
            print("[DF11] No dirty modules found")
            return
        
        print(f"[DF11] Found {dirty_found} dirty modules, forcing TensorManager re-decode...")
        
        # Clear TensorManager's cached tensors
        # This is the key - the weights are views into this buffer
        if hasattr(TensorManager, '_tensors') and TensorManager._tensors:
            # Get the device we need to clear
            target_device = device_to if device_to else torch.device('cuda:0')
            if target_device in TensorManager._tensors:
                # Delete the cached tensor for this device
                del TensorManager._tensors[target_device]
                print(f"[DF11] Cleared TensorManager cache for {target_device}")
            
            torch.cuda.empty_cache()
        
        # Clear dirty flags and comfy_patched_weights to force re-load
        for n, m in dirty_modules:
            m._df11_lora_modified = False
            if hasattr(m, 'comfy_patched_weights'):
                m.comfy_patched_weights = False
            reset_count += 1
        
        # Also clear comfy_patched_weights on ALL modules to force full reload
        for n, m in self.model.named_modules():
            if hasattr(m, 'comfy_patched_weights'):
                m.comfy_patched_weights = False
        
        print(f"[DF11] Reset {reset_count} modules, cleared TensorManager cache")

    def _patch_state_dict(self, state_dict_func):
        lora_loading_functions = {"model_lora_keys_unet", "add_patches"} 
        from . import state_dict_shapes
        all_keys = set()
        for name in state_dict_shapes.__all__:
            keys_dict = getattr(state_dict_shapes, name)
            all_keys.update(keys_dict.keys())
        
        # TODO: Refine fake_state_dict to return only model-specific keys instead of all keys
        # from Chroma, Flux, and Z Image. Current approach has significant key overlap between
        # model types (e.g., Flux.1-dev vs Flux.1-schnell) which could cause issues.
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
                    skip = True # skip random weights in non leaf modules 
                    break
            if not skip and (hasattr(module, "comfy_cast_weights") or len(params) > 0):
                loading.append((comfy.model_management.module_size(module), n, module, params))

        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        print(f"[DF11 LOAD] Starting load... Device: {device_to}, Full: {full_load}")
        
        # Get current patch keys
        current_patch_keys = frozenset(self.patches.keys())
        has_lora = len(current_patch_keys) > 0
        
        # Check ACTUAL state from the model, not instance variables
        weights_are_dirty = self._check_weights_dirty()
        
        print(f"[DF11 LOAD] Patch keys: {len(self._last_patch_keys) if self._last_patch_keys else 'None'} -> {len(current_patch_keys)}")
        print(f"[DF11 LOAD] has_lora={has_lora}, weights_are_dirty={weights_are_dirty}")
        
        # CRITICAL: If no LoRA requested but weights are dirty, we MUST reset
        if not has_lora and weights_are_dirty:
            print(f"[DF11 LOAD] *** NO LORA BUT WEIGHTS DIRTY - RESETTING TO BASE ***")
            self._clear_lora_hooks()
            self._reset_weights_to_base(device_to)
        # If LoRA changed, also reset
        elif has_lora and self._last_patch_keys is not None and current_patch_keys != self._last_patch_keys:
            print(f"[DF11 LOAD] *** LORA CHANGED - RESETTING TO BASE ***")
            self._clear_lora_hooks()
            if weights_are_dirty:
                self._reset_weights_to_base(device_to)
        
        # Update tracking
        self._last_patch_keys = current_patch_keys
        
        need_new_hooks = has_lora and len(self.lora_hook_handles) == 0

        with self.use_ejected():
            super().unpatch_hooks()

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
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed 
                            continue
                        
                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                            if need_new_hooks:
                                print(f"[DF11 LOAD] LowVRAM: Registering LoRA hook for {weight_key}")
                                handle = m.register_forward_pre_hook(get_hook_lora(self.patches[weight_key], weight_key))
                                self.lora_hook_handles.append(handle)
                        else:
                            _, set_func, convert_func = get_key_weight(self.model, weight_key)
                            m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                            patch_counter += 1
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
                
                weights_already_loaded = hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True

                # Load/decode weights from DFloat11
                if not weights_already_loaded:
                    for param in params:
                        self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)
                
                # Register hooks ONLY if we need new hooks AND this key has patches
                weight_key = f"{n}.weight"
                if weight_key in self.patches and need_new_hooks:
                    print(f"[DF11 LOAD] Registering LoRA hook for {weight_key}")
                    handle = m.register_forward_pre_hook(get_hook_lora(self.patches[weight_key], weight_key))
                    self.lora_hook_handles.append(handle)
                
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            # Mark that we now have hooks that will modify weights
            if len(self.lora_hook_handles) > 0:
                self._mark_weights_dirty()
                
            print(f"[DF11 LOAD] Total LoRA hooks: {len(self.lora_hook_handles)}, weights_dirty: {self._check_weights_dirty()}")
            
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
