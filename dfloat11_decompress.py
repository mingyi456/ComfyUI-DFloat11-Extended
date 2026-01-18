import cupy as cp
import pkg_resources
import math

import torch

from dfloat11.dfloat11 import TensorManager
import gc

ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')

bytes_per_thread = 8
threads_per_block = (512, )

klein_4b_sizes = {
    "double_blocks.0.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.0.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.0.img_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.0.img_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.0.img_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.0.img_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.0.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.0.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.0.txt_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.0.txt_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.0.txt_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.0.txt_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.1.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.1.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.1.img_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.1.img_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.1.img_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.1.img_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.1.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.1.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.1.txt_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.1.txt_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.1.txt_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.1.txt_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.2.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.2.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.2.img_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.2.img_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.2.img_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.2.img_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.2.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.2.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.2.txt_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.2.txt_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.2.txt_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.2.txt_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.3.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.3.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.3.img_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.3.img_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.3.img_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.3.img_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.3.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.3.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.3.txt_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.3.txt_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.3.txt_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.3.txt_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.4.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.4.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.4.img_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.4.img_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.4.img_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.4.img_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_blocks.4.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.4.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.4.txt_attn.proj.weight" : torch.Size([3072, 3072]),
    "double_blocks.4.txt_attn.qkv.weight" : torch.Size([9216, 3072]),
    "double_blocks.4.txt_mlp.0.weight" : torch.Size([18432, 3072]),
    "double_blocks.4.txt_mlp.2.weight" : torch.Size([3072, 9216]),
    "double_stream_modulation_img.lin.weight" : torch.Size([18432, 3072]),
    "double_stream_modulation_txt.lin.weight" : torch.Size([18432, 3072]),
    "final_layer.adaLN_modulation.1.weight" : torch.Size([6144, 3072]),
    "final_layer.linear.weight" : torch.Size([128, 3072]),
    "img_in.weight" : torch.Size([3072, 128]),
    "single_blocks.0.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.0.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.0.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.0.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.1.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.1.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.1.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.1.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.10.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.10.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.10.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.10.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.11.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.11.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.11.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.11.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.12.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.12.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.12.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.12.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.13.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.13.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.13.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.13.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.14.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.14.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.14.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.14.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.15.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.15.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.15.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.15.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.16.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.16.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.16.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.16.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.17.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.17.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.17.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.17.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.18.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.18.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.18.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.18.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.19.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.19.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.19.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.19.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.2.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.2.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.2.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.2.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.3.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.3.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.3.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.3.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.4.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.4.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.4.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.4.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.5.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.5.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.5.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.5.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.6.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.6.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.6.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.6.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.7.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.7.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.7.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.7.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.8.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.8.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.8.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.8.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.9.linear1.weight" : torch.Size([27648, 3072]),
    "single_blocks.9.linear2.weight" : torch.Size([3072, 12288]),
    "single_blocks.9.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.9.norm.query_norm.scale" : torch.Size([128]),
    "single_stream_modulation.lin.weight" : torch.Size([9216, 3072]),
    "time_in.in_layer.weight" : torch.Size([3072, 256]),
    "time_in.out_layer.weight" : torch.Size([3072, 3072]),
    "txt_in.weight" : torch.Size([3072, 7680]),
}

def decompress_state_dict_flux_2_klein_4b(df11_state_dict):
    bf16_sizes = klein_4b_sizes
    reconstructed_state_dict = {}
    
    pattern_dict_extras = {
        "double_stream_modulation_img.lin" : [],
        "double_stream_modulation_txt.lin" : [],
        "single_stream_modulation.lin" : [],
    }

    double_block_modules = (
        "img_attn.qkv",
        "img_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_attn.qkv",
        "txt_attn.proj",
        "txt_mlp.0",
        "txt_mlp.2",
    )

    single_block_modules = (
        "linear1",
        "linear2",
    )

    
    for compression_block, compressed_modules in pattern_dict_extras.items():
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.weight"]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
        # reconstructed = torch.empty(n_elements, dtype = torch.bfloat16, device = "cuda")
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])
        
        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    for block_num in range(5):
        compression_block = f"double_blocks.{block_num}"
        compressed_modules = double_block_modules
        
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.{module}.weight" for module in compressed_modules]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])
        
        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    
    for block_num in range(20):
        compression_block = f"single_blocks.{block_num}"
        compressed_modules = single_block_modules
        
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.{module}.weight" for module in compressed_modules]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])

        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    uncompressed_keys = bf16_sizes.keys() & df11_state_dict.keys()
    reconstructed_state_dict.update((key, df11_state_dict[key]) for key in uncompressed_keys)
    del df11_state_dict
    
    gc.collect()
    torch.cuda.empty_cache()

    return reconstructed_state_dict


klein_9b_sizes = {
    "double_blocks.0.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.0.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.0.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.0.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.0.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.0.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.0.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.0.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.0.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.0.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.0.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.0.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.1.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.1.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.1.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.1.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.1.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.1.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.1.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.1.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.1.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.1.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.1.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.1.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.2.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.2.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.2.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.2.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.2.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.2.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.2.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.2.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.2.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.2.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.2.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.2.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.3.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.3.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.3.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.3.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.3.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.3.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.3.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.3.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.3.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.3.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.3.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.3.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.4.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.4.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.4.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.4.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.4.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.4.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.4.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.4.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.4.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.4.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.4.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.4.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.5.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.5.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.5.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.5.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.5.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.5.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.5.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.5.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.5.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.5.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.5.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.5.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.6.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.6.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.6.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.6.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.6.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.6.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.6.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.6.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.6.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.6.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.6.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.6.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.7.img_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.7.img_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.7.img_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.7.img_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.7.img_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.7.img_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_blocks.7.txt_attn.norm.key_norm.scale" : torch.Size([128]),
    "double_blocks.7.txt_attn.norm.query_norm.scale" : torch.Size([128]),
    "double_blocks.7.txt_attn.proj.weight" : torch.Size([4096, 4096]),
    "double_blocks.7.txt_attn.qkv.weight" : torch.Size([12288, 4096]),
    "double_blocks.7.txt_mlp.0.weight" : torch.Size([24576, 4096]),
    "double_blocks.7.txt_mlp.2.weight" : torch.Size([4096, 12288]),
    "double_stream_modulation_img.lin.weight" : torch.Size([24576, 4096]),
    "double_stream_modulation_txt.lin.weight" : torch.Size([24576, 4096]),
    "final_layer.adaLN_modulation.1.weight" : torch.Size([8192, 4096]),
    "final_layer.linear.weight" : torch.Size([128, 4096]),
    "img_in.weight" : torch.Size([4096, 128]),
    "single_blocks.0.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.0.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.0.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.0.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.1.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.1.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.1.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.1.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.10.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.10.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.10.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.10.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.11.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.11.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.11.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.11.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.12.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.12.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.12.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.12.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.13.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.13.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.13.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.13.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.14.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.14.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.14.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.14.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.15.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.15.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.15.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.15.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.16.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.16.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.16.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.16.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.17.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.17.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.17.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.17.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.18.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.18.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.18.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.18.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.19.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.19.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.19.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.19.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.2.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.2.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.2.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.2.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.20.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.20.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.20.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.20.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.21.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.21.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.21.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.21.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.22.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.22.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.22.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.22.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.23.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.23.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.23.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.23.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.3.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.3.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.3.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.3.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.4.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.4.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.4.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.4.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.5.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.5.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.5.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.5.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.6.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.6.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.6.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.6.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.7.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.7.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.7.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.7.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.8.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.8.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.8.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.8.norm.query_norm.scale" : torch.Size([128]),
    "single_blocks.9.linear1.weight" : torch.Size([36864, 4096]),
    "single_blocks.9.linear2.weight" : torch.Size([4096, 16384]),
    "single_blocks.9.norm.key_norm.scale" : torch.Size([128]),
    "single_blocks.9.norm.query_norm.scale" : torch.Size([128]),
    "single_stream_modulation.lin.weight" : torch.Size([12288, 4096]),
    "time_in.in_layer.weight" : torch.Size([4096, 256]),
    "time_in.out_layer.weight" : torch.Size([4096, 4096]),
    "txt_in.weight" : torch.Size([4096, 12288]),
}

def decompress_state_dict_flux_2_klein_9b(df11_state_dict):
    bf16_sizes = klein_9b_sizes
    reconstructed_state_dict = {}
    
    pattern_dict_extras = {
        "double_stream_modulation_img.lin" : [],
        "double_stream_modulation_txt.lin" : [],
        "single_stream_modulation.lin" : [],
    }

    double_block_modules = (
        "img_attn.qkv",
        "img_attn.proj",
        "img_mlp.0",
        "img_mlp.2",
        "txt_attn.qkv",
        "txt_attn.proj",
        "txt_mlp.0",
        "txt_mlp.2",
    )

    single_block_modules = (
        "linear1",
        "linear2",
    )

    
    for compression_block, compressed_modules in pattern_dict_extras.items():
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.weight"]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
        # reconstructed = torch.empty(n_elements, dtype = torch.bfloat16, device = "cuda")
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])
        
        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    for block_num in range(8):
        compression_block = f"double_blocks.{block_num}"
        compressed_modules = double_block_modules
        
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.{module}.weight" for module in compressed_modules]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])
        
        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    
    for block_num in range(24):
        compression_block = f"single_blocks.{block_num}"
        compressed_modules = single_block_modules
        
        cuda_device = torch.device("cuda")
        
        target_modules = [f"{compression_block}.{module}.weight" for module in compressed_modules]
        
        luts = df11_state_dict.pop(f"{compression_block}.luts").to("cuda")
        encoded_exponent = df11_state_dict.pop(f"{compression_block}.encoded_exponent").to("cuda")
        sign_mantissa = df11_state_dict.pop(f"{compression_block}.sign_mantissa").to("cuda")
        output_positions = df11_state_dict.pop(f"{compression_block}.output_positions").to("cuda")
        gaps = df11_state_dict.pop(f"{compression_block}.gaps").to("cuda")
        split_positions = df11_state_dict.pop(f"{compression_block}.split_positions")
    
        n_elements = sign_mantissa.numel()
        n_bytes = encoded_exponent.numel()
        n_luts = luts.shape[0]
    
        reconstructed = TensorManager.allocate_bfloat16(cuda_device, n_elements)
    
        output_positions_np = output_positions.cpu().view(torch.uint32).numpy()
        shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
        # print(f'Using {shared_mem_size} bytes of shared memory.')
    
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )
        
        with cp.cuda.Device(cuda_device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                luts.data_ptr(),
                encoded_exponent.data_ptr(),
                sign_mantissa.data_ptr(),
                output_positions.data_ptr(),
                gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])
    
        reconstructed_weights = torch.tensor_split(reconstructed, split_positions)
    
        for target_module, reconstructed_weight in zip(target_modules, reconstructed_weights):
            reconstructed_state_dict[target_module] = reconstructed_weight.cpu().view(bf16_sizes[target_module])

        del luts
        del encoded_exponent
        del sign_mantissa
        del output_positions
        del gaps
        del split_positions
    
    uncompressed_keys = bf16_sizes.keys() & df11_state_dict.keys()
    reconstructed_state_dict.update((key, df11_state_dict[key]) for key in uncompressed_keys)
    del df11_state_dict
    
    gc.collect()
    torch.cuda.empty_cache()

    return reconstructed_state_dict

decompress_state_dict_func_map = {
    "Flux.2-Klein-4B": decompress_state_dict_flux_2_klein_4b,
    "Flux.2-Klein-9B": decompress_state_dict_flux_2_klein_9b,
}



