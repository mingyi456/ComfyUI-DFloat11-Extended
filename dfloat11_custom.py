import comfy
import folder_paths
import torch
import torch.nn as nn

import os

import logging
import inspect
from comfy.model_patcher import LowVramPatch, move_weight_functions, wipe_lowvram_weight, get_key_weight, string_to_seed
from comfy.patcher_extension import CallbacksMP

from comfy.quant_ops import QuantizedTensor

from .state_dict_shapes import state_dict_mapping

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
        device = weight.device
        
        # Get reusable fp16 buffer
        fp16_buffer = CastBufferManager.get_float16_buffer(device, n_elements).view(original_shape)
        fp16_buffer.copy_(weight) # `Tensor.copy_()` handles typecasting automatically
        
        # Calculate LoRA - this mutates the `fp16_buffer` tensor in place
        try:
            comfy.lora.calculate_weight(patch_list, fp16_buffer, key)
        except Exception as e:
            print(f"[LORA HOOK ERROR] Failed to calculate weight for {key}: {e}")
            raise e
        
        weight.copy_(fp16_buffer)
            
    return lora_hook

LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR = 2

def low_vram_patch_estimate_vram(model, key):
    weight, set_func, convert_func = get_key_weight_df11(model, key)
    if weight is None:
        return 0
    model_dtype = getattr(model, "manual_cast_dtype", torch.float32)
    if model_dtype is None:
        model_dtype = weight.dtype

    return weight.numel() * model_dtype.itemsize * LOWVRAM_PATCH_ESTIMATE_MATH_FACTOR

def parent_is_offloaded(model, key):
    parts = key.split(".")
    
    parent = model
    for part in parts:
        if hasattr(parent, "offloaded_tensors"):
            return True
        parent = getattr(parent, part)
    return False

def get_key_weight_df11(model, key):
    #  key = 'diffusion_model.noise_refiner.0.attention.qkv.weight'
    set_func = None
    convert_func = None
    op_keys = key.rsplit('.', 1)
    if len(op_keys) < 2:
        weight = comfy.utils.get_attr(model, key)
    else:
        op = comfy.utils.get_attr(model, op_keys[0]) #  op_keys[0] = 'diffusion_model.noise_refiner.0.attention.qkv'
        set_func = getattr(op, "set_{}".format(op_keys[1]), None)
        convert_func = getattr(op, "convert_{}".format(op_keys[1]), None)

        weight = getattr(op, op_keys[1], None)
        if weight is None:
            pass
        if convert_func is not None:
            weight = comfy.utils.get_attr(model, key)

    return weight, set_func, convert_func

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

        self._patch_state_dict()
        # List to keep track of PyTorch hooks, currently unused but retained just-in-case
        self.lora_hook_handles = []

    def _patch_state_dict(self):
        if hasattr(self.model.state_dict, "patched_for_lora"):
            return
        state_dict_func = self.model.state_dict
        df11_type = type(self.model.model_config).__name__
        if df11_type in state_dict_mapping:
            logging.info(f"[DF11] Supported df11_type for LoRA: {df11_type}")
            fake_keys = state_dict_mapping[df11_type].keys()
            fake_state_dict = {f"diffusion_model.{key}": None for key in fake_keys}
        else:
            logging.info(f"[DF11] Unsupported df11_type for LoRA: {df11_type}")
            fake_state_dict = state_dict_func()
        
        lora_loading_functions = {"model_lora_keys_unet", "add_patches"} 
        
        def new_state_dict_func():
            call_stack = inspect.stack()
            caller_function = call_stack[1].function
            del call_stack
            if caller_function in lora_loading_functions:
                return fake_state_dict
            return state_dict_func()
        self.model.state_dict = new_state_dict_func
        self.model.state_dict.patched_for_lora = True

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

    def _load_list(self):
        print("[DF11] _load_list_v1()")
        # This uses the old implementation of `_load_list()`, 
        # because the new implementation bricks due to accessing missing `.weight` attributes
        # TODO: Update this method to align with current ComfyUI's `_load_list()` structure, 
        # taking into account DF11
        loading = []
        for module_name, module in self.model.named_modules():
            params = []
            skip = False
            for name, param in module.named_parameters(recurse=False):
                params.append(name)
            for name, param in module.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(module, "comfy_cast_weights") or len(params) > 0):
                loading.append((comfy.model_management.module_size(module), module_name, module, params))

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
                # If the module is not compressed via DF11, weight and bias keys will both exist in `params` and thus patching will occur normally
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


    def _load_list_v1_5(self):
        print("[DF11] _load_list_v1_5()")
        # This uses the old implementation of `_load_list()`, 
        # because the new implementation bricks due to accessing missing `.weight` attributes
        # TODO: Update this method to align with current ComfyUI's `_load_list()` structure, 
        # taking into account DF11
        loading = []
        for module_name, module in self.model.named_modules():
            params = []
            skip = False
            for name, param in module.named_parameters(recurse=False):
                params.append(name)
            for name, param in module.named_parameters(recurse=True):
                if name not in params:
                    skip = True # skip random weights in non leaf modules
                    break

            if not skip and (hasattr(module, "comfy_cast_weights") or len(params) > 0):
                module_size = comfy.model_management.module_size(module)
                if (module_size == 0) and isinstance(module, nn.Linear):
                    if not parent_is_offloaded(self.model, module_name):
                        module_size += int(module.in_features * module.out_features * 1.4)
                loading.append((module_size, module_name, module, params))

        return loading


    def _load_list_v2(self):
        print("[DF11] _load_list_v2()")
        loading = []
        for module_name, module in self.model.named_modules():
            params = []
            for param_name, param in module.named_parameters(recurse=False):
                params.append(param_name)
            skip = any(param_name not in params for param_name, param in module.named_parameters(recurse=True))
            
            if not skip and (hasattr(module, "comfy_cast_weights") or len(params) > 0):
                module_mem = comfy.model_management.module_size(module) 
                # DF11 modules are usually compressed at the block-level, so `module_mem` returns 0
                if (module_mem == 0) and isinstance(module, nn.Linear):
                    if not parent_is_offloaded(self.model, module_name):
                        module_mem += int(module.in_features * module.out_features * 1.4)
                module_offload_mem = module_mem
                if hasattr(module, "comfy_cast_weights"):
                    def check_module_offload_mem(key, module):
                        if key in self.patches:
                            return low_vram_patch_estimate_vram(self.model, key)
                        model_dtype = getattr(self.model, "manual_cast_dtype", None)
                        weight, _, _ = get_key_weight_df11(self.model, key)
                        if model_dtype is None or weight is None:
                            return 0
                        if (weight.dtype != model_dtype or isinstance(weight, QuantizedTensor)):
                            return weight.numel() * model_dtype.itemsize
                        return 0
                    module_offload_mem += check_module_offload_mem("{}.weight".format(module_name), module)
                    module_offload_mem += check_module_offload_mem("{}.bias".format(module_name), module)
                loading.append((module_offload_mem, module_mem, module_name, module, params))
        return loading

    def load_v2(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
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

            # super().unpatch_hooks()

            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            lowvram_mem_counter = 0
            loading = self._load_list()

            load_completely = []
            offloaded = []
            offload_buffer = 0
            loading.sort(reverse=True)
            for i, x in enumerate(loading):
                module_offload_mem, module_mem, n, m, params = x

                lowvram_weight = False

                potential_offload = max(offload_buffer, module_offload_mem + sum([ x1[1] for x1 in loading[i+1:i+1+comfy.model_management.NUM_STREAMS]]))
                lowvram_fits = mem_counter + module_mem + potential_offload < lowvram_model_memory

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if not lowvram_fits:
                        offload_buffer = potential_offload
                        lowvram_weight = True
                        lowvram_counter += 1
                        lowvram_mem_counter += module_mem
                        if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                if lowvram_weight:
                    logging.warning("[DF11] VRAM appears to be insufficient according to ComfyUI, this will likely cause errors due to conflicts with Comfy's native block swapping mechanism and DF11 weights. Consider enabling `cpu_offload` and increasing `cpu_offload_blocks` (available via the \"DFloat11 Model Loader (Advanced)\" node")
                    if hasattr(m, "comfy_cast_weights"):
                        m.weight_function = []
                        m.bias_function = []

                    if weight_key in self.patches:
                        if force_patch_weights:
                            self.patch_weight_to_device(weight_key)
                        else:
                            # This will fail for DF11 compressed modules, but fixing this is useless without fixing `self.pin_weight_to_device()` as well
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
                    offloaded.append((module_mem, n, m, params))
                else:
                    if hasattr(m, "comfy_cast_weights"):
                        wipe_lowvram_weight(m)

                    if full_load or lowvram_fits:
                        mem_counter += module_mem
                        load_completely.append((module_mem, n, m, params))
                    else:
                        offload_buffer = potential_offload

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
                    key = "{}.{}".format(n, param)
                    self.unpin_weight(key)
                    self.patch_weight_to_device(key, device_to=device_to)

                weight_key = f"{n}.weight"
                if need_new_hooks and weight_key in self.patches:
                    handle = m.register_forward_pre_hook(get_hook_lora(self.patches[weight_key], weight_key))
                    self.lora_hook_handles.append(handle)

                if comfy.model_management.is_device_cuda(device_to):
                    torch.cuda.synchronize()

                logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
                m.comfy_patched_weights = True

            for x in load_completely:
                x[2].to(device_to)

            for x in offloaded:
                n = x[1]
                params = x[3]
                for param in params:
                    self.pin_weight_to_device("{}.{}".format(n, param))

            if lowvram_counter > 0:
                logging.info("loaded partially; {:.2f} MB usable, {:.2f} MB loaded, {:.2f} MB offloaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), lowvram_mem_counter / (1024 * 1024), offload_buffer / (1024 * 1024), patch_counter))
                self.model.model_lowvram = True
            else:
                logging.info("loaded completely; {:.2f} MB usable, {:.2f} MB loaded, full load: {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
                self.model.model_lowvram = False
                if full_load:
                    self.model.to(device_to)
                    mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.model_offload_buffer_memory = offload_buffer
            self.model.current_weight_patches_uuid = self.patches_uuid

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)
            
            logging.info(f"[DF11] Load complete. LoRA hooks: {len(self.lora_hook_handles)}, has_patches: {has_patches}")

    def partially_unload_v2(self, device_to, memory_to_free=0, force_patch_weights=False):
        with self.use_ejected():
            hooks_unpatched = False
            memory_freed = 0
            patch_counter = 0
            unload_list = self._load_list()
            unload_list.sort()

            offload_buffer = self.model.model_offload_buffer_memory
            if len(unload_list) > 0:
                NS = comfy.model_management.NUM_STREAMS
                offload_weight_factor = [ min(offload_buffer / (NS + 1), unload_list[0][1]) ] * NS

            for unload in unload_list:
                if memory_to_free + offload_buffer - self.model.model_offload_buffer_memory < memory_freed:
                    break
                module_offload_mem, module_mem, n, m, params = unload

                potential_offload = module_offload_mem + sum(offload_weight_factor)

                lowvram_possible = hasattr(m, "comfy_cast_weights")
                if hasattr(m, "comfy_patched_weights") and m.comfy_patched_weights == True:
                    move_weight = True
                    for param in params:
                        key = "{}.{}".format(n, param)
                        bk = self.backup.get(key, None)
                        if bk is not None:
                            if not lowvram_possible:
                                move_weight = False
                                break

                            if not hooks_unpatched:
                                self.unpatch_hooks()
                                hooks_unpatched = True

                            if bk.inplace_update:
                                comfy.utils.copy_to_param(self.model, key, bk.weight)
                            else:
                                comfy.utils.set_attr_param(self.model, key, bk.weight)
                            self.backup.pop(key)

                    weight_key = "{}.weight".format(n)
                    bias_key = "{}.bias".format(n)
                    if move_weight:
                        cast_weight = self.force_cast_weights
                        m.to(device_to)
                        module_mem += move_weight_functions(m, device_to)
                        if lowvram_possible:
                            if weight_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(weight_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, weight_key)
                                    m.weight_function.append(LowVramPatch(weight_key, self.patches, convert_func, set_func))
                                    patch_counter += 1
                            if bias_key in self.patches:
                                if force_patch_weights:
                                    self.patch_weight_to_device(bias_key)
                                else:
                                    _, set_func, convert_func = get_key_weight(self.model, bias_key)
                                    m.bias_function.append(LowVramPatch(bias_key, self.patches, convert_func, set_func))
                                    patch_counter += 1
                            cast_weight = True

                        if cast_weight and hasattr(m, "comfy_cast_weights"):
                            m.prev_comfy_cast_weights = m.comfy_cast_weights
                            m.comfy_cast_weights = True
                        m.comfy_patched_weights = False
                        memory_freed += module_mem
                        offload_buffer = max(offload_buffer, potential_offload)
                        offload_weight_factor.append(module_mem)
                        offload_weight_factor.pop(0)
                        logging.debug("freed {}".format(n))

                        for param in params:
                            self.pin_weight_to_device("{}.{}".format(n, param))


            self.model.model_lowvram = True
            self.model.lowvram_patch_counter += patch_counter
            self.model.model_loaded_weight_memory -= memory_freed
            self.model.model_offload_buffer_memory = offload_buffer
            logging.info("Unloaded partially: {:.2f} MB freed, {:.2f} MB remains loaded, {:.2f} MB buffer reserved, lowvram patches: {}".format(memory_freed / (1024 * 1024), self.model.model_loaded_weight_memory / (1024 * 1024), offload_buffer / (1024 * 1024), self.model.lowvram_patch_counter))
            return memory_freed

    def patch_loading_methods(self, load_version = None):

        if load_version is None and not hasattr(self, "load_version"):
            return

        if not hasattr(self, "_load_list_v1"):
            self._load_list_v1 = self._load_list
            self.load_v1 = self.load
            self.partially_unload_v1 = self.partially_unload

        if load_version is not None:
            self.load_version = load_version

        if self.load_version == "v1":
            self._load_list = self._load_list_v1
            self.load = self.load_v1
            self.partially_unload = self.partially_unload_v1
        
        elif self.load_version == "v1.5":
            self._load_list = self._load_list_v1_5
            self.load = self.load_v1
            self.partially_unload = self.partially_unload_v1
        
        elif self.load_version == "v2":
            self._load_list = self._load_list_v2
            self.load = self.load_v2
            self.partially_unload = self.partially_unload_v2

    def clone(self):
        new_model_patcher = super().clone()
        
        if hasattr(self, "load_version"):
            new_model_patcher.patch_loading_methods(self.load_version)
        
        return new_model_patcher

