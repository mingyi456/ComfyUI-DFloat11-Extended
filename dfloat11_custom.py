import comfy
import folder_paths
import torch
import torch.nn as nn

import os

import cupy as cp
import pkg_resources
import re
import math

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

from .convert_fixed_tensors import convert_diffusers_to_comfyui_flux

ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')


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