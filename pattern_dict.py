MODEL_TO_PATTERN_DICT = {
    "Flux": {
        r"double_blocks\.\d+": (
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
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
            "modulation.lin",
        ),
    },
    "FluxSchnell": {
        r"double_blocks\.\d+": (
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
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
            "modulation.lin",
        ),
    },
    "Chroma": {
        r"distilled_guidance_layer\.layers\.\d+": (
            "in_layer",
            "out_layer"        
        ),
        r"double_blocks\.\d+": (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ),
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
        ),
    },
    "ChromaRadiance": {
        r"distilled_guidance_layer\.layers\.\d+": (
            "in_layer",
            "out_layer"        
        ),
        r"double_blocks\.\d+": (
            "img_attn.qkv",
            "img_attn.proj",
            "img_mlp.0",
            "img_mlp.2",
            "txt_attn.qkv",
            "txt_attn.proj",
            "txt_mlp.0",
            "txt_mlp.2",
        ),
        r"single_blocks\.\d+": (
            "linear1",
            "linear2",
        ),
        r"nerf_blocks\.\d+": (
            "param_generator", 
        )
    }, 
    "CosmosT2IPredict2": {
        "t_embedder\.1": (
            "linear_1",
            "linear_2",
        ),
        r"blocks\.\d+": (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.output_proj",
            "cross_attn.q_proj",
            "cross_attn.k_proj",
            "cross_attn.v_proj",
            "cross_attn.output_proj",
            "mlp.layer1",
            "mlp.layer2",
            "adaln_modulation_self_attn.1",
            "adaln_modulation_self_attn.2",
            "adaln_modulation_cross_attn.1",
            "adaln_modulation_cross_attn.2",
            "adaln_modulation_mlp.1",
            "adaln_modulation_mlp.2",
        )
    },
    "Lumina2": {
        r"noise_refiner\.\d+": (
            "attention.qkv",
            "attention.out",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "adaLN_modulation.1"
        ),
        r"context_refiner\.\d+": (
            "attention.qkv",
            "attention.out",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        ),   
        r"layers\.\d+": (
            "attention.qkv",
            "attention.out",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "adaLN_modulation.1"
        )
    },
    "SDXL": {
        r"time_embed" : (
            "0",
            "2",
        ),
        r"label_emb.0" : (
            "0",
            "2",
        ),
        
        r"input_blocks\.[12]\.0" : (
            "emb_layers.1",
        ),

        r"input_blocks\.4\.0" : (
            "emb_layers.1",
        ),

        r"input_blocks\.4\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        
        r"input_blocks\.5\.0" : (
            "emb_layers.1",
        ),

        r"input_blocks\.5\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        
        
        r"input_blocks\.7\.0" : (
            "emb_layers.1",
        ),
        
        r"input_blocks\.7\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),

        r"input_blocks\.8\.0" : (
            "emb_layers.1",
        ),

        r"input_blocks\.8\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),


        r"middle_block\.0" : (
            "emb_layers.1",
        ),
        r"middle_block\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        r"middle_block\.2" : (
            "emb_layers.1",
        ),


        r"output_blocks\.[01]\.0" : (
            "emb_layers.1",
        ),
        r"output_blocks\.[01]\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        r"output_blocks\.2\.0" : (
            "emb_layers.1",
        ),
        r"output_blocks\.2\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),

        r"output_blocks\.3\.0" : (
            "emb_layers.1",
        ),
        r"output_blocks\.3\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        r"output_blocks\.4\.0" : (
            "emb_layers.1",
        ),
        r"output_blocks\.4\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),
        r"output_blocks\.5\.0" : (
            "emb_layers.1",
        ),
        r"output_blocks\.5\.1\.transformer_blocks\.\d+" : (
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ),

        r"output_blocks\.[678]\.0" : (
            "emb_layers.1",
        ),
    },
}
