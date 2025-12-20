import comfy
import folder_paths
import torch
import torch.nn as nn

import os

import logging
import inspect
from comfy.model_patcher import LowVramPatch, move_weight_functions, wipe_lowvram_weight, get_key_weight, string_to_seed
from comfy.patcher_extension import CallbacksMP

def get_hook_lora(patch_list, key):
    def lora_hook(module, input):
        new_weight = comfy.lora.calculate_weight(patch_list, module.weight, key)
        module.weight = comfy.float.stochastic_rounding(new_weight, module.weight.dtype, seed=string_to_seed(key))
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
    
    def unpatch_hooks(self):
        # 1. Run standard ComfyUI unpatching
        super().unpatch_hooks()
        
        # 2. Explicitly remove our custom LoRA hooks
        if hasattr(self, "lora_hook_handles"):
            for hook in self.lora_hook_handles:
                hook.remove()
            self.lora_hook_handles.clear()

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
        with self.use_ejected():
            self.unpatch_hooks()
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
                            # Register Hook & Store Handle (moved here - no longer depends on bias)
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
                params = x[3] # ['weight', 'bias']
                if hasattr(m, "comfy_patched_weights"):
                    if m.comfy_patched_weights == True:
                        continue

                for param in params:
                    self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)
                
                # Apply LoRA Hook if patches exist for this module's weight
                # We do this OUTSIDE the param loop to ensure it happens once per module
                # and doesn't depend on "bias" existing.
                weight_key = f"{n}.weight"
                if weight_key in self.patches:
                     handle = m.register_forward_pre_hook(get_hook_lora(self.patches[weight_key], weight_key))
                     self.lora_hook_handles.append(handle)

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
