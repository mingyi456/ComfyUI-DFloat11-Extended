from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from itertools import chain
import gc

double_map = {
	"encoded_exponent" : "encoded_exponent",
	"gaps" : "gaps",
	"luts" : "luts",
	"output_positions" : "output_positions",
	"sign_mantissa" : "sign_mantissa",
	"split_positions" : "split_positions",
	"img_attn.norm.key_norm.scale" : "attn.norm_k.weight",
	"img_attn.norm.query_norm.scale" : "attn.norm_q.weight",
	"img_attn.proj.bias" : "attn.to_out.0.bias",
	# "img_attn.qkv.bias" : ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"),
	"img_mlp.0.bias" : "ff.net.0.proj.bias",
	"img_mlp.2.bias" : "ff.net.2.bias",
	"img_mod.lin.bias" : "norm1.linear.bias",
	"txt_attn.norm.key_norm.scale" : "attn.norm_added_k.weight",
	"txt_attn.norm.query_norm.scale" : "attn.norm_added_q.weight",
	"txt_attn.proj.bias" : "attn.to_add_out.bias",
	# "txt_attn.qkv.bias" : ("attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"),
	"txt_mlp.0.bias" : "ff_context.net.0.proj.bias",
	"txt_mlp.2.bias" : "ff_context.net.2.bias",
	"txt_mod.lin.bias" : "norm1_context.linear.bias"
}

double_map_multi = {
	"img_attn.qkv.bias" : ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"),
	"txt_attn.qkv.bias" : ("attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias")
}

single_map = {
	"encoded_exponent" : "encoded_exponent",
	"gaps" : "gaps",
	"luts" : "luts",
	"output_positions" : "output_positions",
	"sign_mantissa" : "sign_mantissa",
	"split_positions" : "split_positions",
	# "linear1.bias" : ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"),
	"linear2.bias" : "proj_out.bias",
	"modulation.lin.bias" : "norm.linear.bias",
	"norm.key_norm.scale" : "attn.norm_k.weight",
	"norm.query_norm.scale" : "attn.norm_q.weight",
}

single_map_multi = {"linear1.bias" : ("attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias")}

extras_map = {
	# "final_layer.adaLN_modulation.1.bias" : "norm_out.linear.bias",
	# "final_layer.adaLN_modulation.1.weight" : "norm_out.linear.weight",
	"final_layer.linear.bias" : "proj_out.bias",
	"final_layer.linear.weight" : "proj_out.weight",
	"img_in.bias" : "x_embedder.bias",
	"img_in.weight" : "x_embedder.weight",
	"time_in.in_layer.bias" : "time_text_embed.timestep_embedder.linear_1.bias",
	"time_in.in_layer.weight" : "time_text_embed.timestep_embedder.linear_1.weight",
	"time_in.out_layer.bias" : "time_text_embed.timestep_embedder.linear_2.bias",
	"time_in.out_layer.weight" : "time_text_embed.timestep_embedder.linear_2.weight",
	"txt_in.bias" : "context_embedder.bias",
	"txt_in.weight" : "context_embedder.weight",
	"vector_in.in_layer.bias" : "time_text_embed.text_embedder.linear_1.bias",
	"vector_in.in_layer.weight" : "time_text_embed.text_embedder.linear_1.weight",
	"vector_in.out_layer.bias" : "time_text_embed.text_embedder.linear_2.bias",
	"vector_in.out_layer.weight" : "time_text_embed.text_embedder.linear_2.weight",
}

def swap_scale_shift(weight):
	shift, scale = weight.chunk(2, dim=0)
	new_weight = torch.cat([scale, shift], dim=0)
	return new_weight

extras_map_special = {
	"final_layer.adaLN_modulation.1.bias" : "norm_out.linear.bias",
	"final_layer.adaLN_modulation.1.weight" : "norm_out.linear.weight",
}

extras_map_optional = {
	"guidance_in.in_layer.bias" : "time_text_embed.guidance_embedder.linear_1.bias",
	"guidance_in.in_layer.weight" : "time_text_embed.guidance_embedder.linear_1.weight",
	"guidance_in.out_layer.bias" : "time_text_embed.guidance_embedder.linear_2.bias",
	"guidance_in.out_layer.weight" : "time_text_embed.guidance_embedder.linear_2.weight",
}

def convert_diffusers_transformer_blocks_to_comfyui_double_blocks(diffusers_state_dict, block_num):
	comfyui_state_dict = {}
	double_block_prefix_diffusers = "transformer_blocks."
	double_block_prefix_comfyui = "double_blocks."
	
	for comfyui_suffix, diffusers_suffix in double_map.items():
		comfyui_state_dict[f"{double_block_prefix_comfyui}{block_num}.{comfyui_suffix}"] = diffusers_state_dict.pop(f"{double_block_prefix_diffusers}{block_num}.{diffusers_suffix}")

	for comfyui_suffix, list_diffusers_suffix in double_map_multi.items():
		tensor_comfyui = torch.cat([diffusers_state_dict[f"{double_block_prefix_diffusers}{block_num}.{suffix}"] for suffix in list_diffusers_suffix])
		comfyui_state_dict[f"{double_block_prefix_comfyui}{block_num}.{comfyui_suffix}"] = tensor_comfyui
		for suffix in list_diffusers_suffix:
			temp = diffusers_state_dict.pop(f"{double_block_prefix_diffusers}{block_num}.{suffix}")
			del temp

	return comfyui_state_dict

def convert_diffusers_single_transformer_blocks_to_comfyui_single_blocks(diffusers_state_dict, block_num):
	comfyui_state_dict = {}
	single_block_prefix_diffusers = "single_transformer_blocks."
	single_block_prefix_comfyui = "single_blocks."
	
	for comfyui_suffix, diffusers_suffix in single_map.items():
		comfyui_state_dict[f"{single_block_prefix_comfyui}{block_num}.{comfyui_suffix}"] = diffusers_state_dict.pop(f"{single_block_prefix_diffusers}{block_num}.{diffusers_suffix}")

	for comfyui_suffix, list_diffusers_suffix in single_map_multi.items():
		tensor_comfyui = torch.cat([diffusers_state_dict[f"{single_block_prefix_diffusers}{block_num}.{suffix}"] for suffix in list_diffusers_suffix])
		comfyui_state_dict[f"{single_block_prefix_comfyui}{block_num}.{comfyui_suffix}"] = tensor_comfyui
		for suffix in list_diffusers_suffix:
			temp = diffusers_state_dict.pop(f"{single_block_prefix_diffusers}{block_num}.{suffix}")
			del temp

	return comfyui_state_dict

def convert_diffusers_extras_to_comfyui_extras(diffusers_state_dict):
	comfyui_state_dict = {}

	for comfyui_key, diffusers_key in extras_map.items():
		comfyui_state_dict[comfyui_key] = diffusers_state_dict.pop(diffusers_key)

	for comfyui_key, diffusers_key in extras_map_optional.items():
		if diffusers_key not in diffusers_state_dict:
			break
		comfyui_state_dict[comfyui_key] = diffusers_state_dict.pop(diffusers_key)

	for comfyui_key, diffusers_key in extras_map_special.items():
		comfyui_state_dict[comfyui_key] = swap_scale_shift(diffusers_state_dict.pop(diffusers_key))
	
	return comfyui_state_dict



def convert_diffusers_to_comfyui_flux(all_tensors_diffusers):

	all_keys_diffusers = all_tensors_diffusers.keys()

	# expected_tensor_set = set(chain(double_map.values(), chain.from_iterable(double_map_multi.values()), single_map.values(), chain.from_iterable(single_map_multi.values()), extras_map.values()))
	# print(expected_tensor_set)
	
	double_block_prefix = "transformer_blocks."
	single_block_prefix = "single_transformer_blocks."
	
	num_double_blocks = max(int(i.removeprefix(double_block_prefix).split(".", maxsplit = 1)[0])  for i in filter(lambda key : key.startswith(double_block_prefix), all_keys_diffusers)) + 1
	# print(num_double_blocks)
	
	num_single_blocks = max(int(i.removeprefix(single_block_prefix).split(".", maxsplit = 1)[0])  for i in filter(lambda key : key.startswith(single_block_prefix), all_keys_diffusers)) + 1
	# print(num_single_blocks)

	extras = [key for key in all_keys_diffusers if (not key.startswith(double_block_prefix) and not key.startswith(single_block_prefix))]
	# print(extras)
	
	with torch.inference_mode():
	
		double_block_tensors = {}
		for i in range(num_double_blocks):
			double_block_tensors |= convert_diffusers_transformer_blocks_to_comfyui_double_blocks(all_tensors_diffusers, i)

		single_block_tensors = {}
		for i in range(num_single_blocks):
			single_block_tensors |= convert_diffusers_single_transformer_blocks_to_comfyui_single_blocks(all_tensors_diffusers, i)

		extra_tensors = convert_diffusers_extras_to_comfyui_extras(all_tensors_diffusers)
	
	# print(f"Leftover keys : {all_tensors_diffusers.keys()}")

	gc.collect()
	torch.cuda.empty_cache()


	return double_block_tensors | single_block_tensors | extra_tensors