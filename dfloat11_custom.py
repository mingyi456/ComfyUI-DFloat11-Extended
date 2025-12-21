import comfy
import folder_paths
import torch
import torch.nn as nn

import os

import logging
import inspect
from comfy.model_patcher import LowVramPatch, move_weight_functions, wipe_lowvram_weight, get_key_weight, string_to_seed
from comfy.patcher_extension import CallbacksMP
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
    def lora_hook(module, input):
        weight = module.weight  # bfloat16 view into TensorManager buffer
        original_shape = weight.shape
        n_elements = weight.numel()
        
        # Get reusable fp16 buffer
        fp16_buffer = CastBufferManager.get_float16_buffer(weight.device, n_elements).view(original_shape)
        fp16_buffer.copy_(weight)
        
        # Calculate LoRA - this creates a new tensor (unavoidable)
        try:
            comfy.lora.calculate_weight(patch_list, fp16_buffer, key)
        except Exception as e:
            print(f"[LORA HOOK ERROR] Failed to calculate weight for {key}: {e}")
            raise e
        
        # Copy directly back into the ORIGINAL DFloat11 buffer (in-place)
        weight.copy_(fp16_buffer)
            
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
    
    def _clear_all_lora_hooks_from_model(self):
        """Remove ALL lora_hook forward pre-hooks from the model - not just ones we have handles for"""
        removed = 0
        for n, m in self.model.named_modules():
            if hasattr(m, '_forward_pre_hooks'):
                hooks_to_remove = []
                for hook_id, hook_fn in list(m._forward_pre_hooks.items()):
                    fn_name = getattr(hook_fn, '__name__', '') + getattr(hook_fn, '__qualname__', '')
                    if 'lora_hook' in fn_name:
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    del m._forward_pre_hooks[hook_id]
                    removed += 1
        self.lora_hook_handles.clear()
        if removed > 0:
            logging.debug(f"[DF11] Cleared {removed} LoRA hooks from model")
        return removed

    def _reset_patched_weights(self):
        """Clear comfy_patched_weights flags to force re-decode on next load"""
        for n, m in self.model.named_modules():
            if hasattr(m, 'comfy_patched_weights'):
                m.comfy_patched_weights = False

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            # KEY: Check if patches changed using the MODEL's stored uuid (like official ComfyUI)
            patches_changed = self.model.current_weight_patches_uuid != self.patches_uuid
            
            if patches_changed:
                logging.info(f"[DF11] Patches changed, clearing LoRA hooks and resetting weights")
                self._clear_all_lora_hooks_from_model()
                self._reset_patched_weights()
            
            # Determine if we need to register new hooks
            has_patches = len(self.patches) > 0
            need_new_hooks = has_patches and patches_changed

            super().unpatch_hooks()

            mem_counter = 0
            loading = self._load_list()

            load_completely = []
            loading.sort(reverse=True)
            for x in loading:
                n = x[1]
                m = x[2]
                params = x[3]
                module_mem = x[0]

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if hasattr(m, "comfy_cast_weights"):
                    wipe_lowvram_weight(m)

                if full_load or mem_counter + module_mem < lowvram_model_memory:
                    mem_counter += module_mem
                    load_completely.append((module_mem, n, m, params))

                if hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    if not hasattr(m, 'weight_function'):
                        m.weight_function = []
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    if not hasattr(m, 'bias_function'):
                        m.bias_function = []
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            load_completely.sort(reverse=True)
            for x in load_completely:
                n = x[1]
                m = x[2]
                params = x[3]

                # Skip if already loaded and patches haven't changed
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights and not patches_changed:
                    continue

                # Trigger weight loading (DFloat11 decode happens here or on first forward)
                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

                # Register LoRA hooks ONLY if we need new ones
                weight_key = f"{n}.weight"
                if need_new_hooks and weight_key in self.patches:
                    handle = m.register_forward_pre_hook(get_hook_lora(self.patches[weight_key], weight_key))
                    self.lora_hook_handles.append(handle)

                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

            # Update model state (this is what ComfyUI checks next time)
            self.model.model_lowvram = False
            self.model.lowvram_patch_counter = 0
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.current_weight_patches_uuid = self.patches_uuid  # KEY: Store on model

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

            logging.info(f"[DF11] Load complete. LoRA hooks: {len(self.lora_hook_handles)}, has_patches: {has_patches}")
