pattern_dict = {
    r"text_embedding" : (
        "0",
        "2",
    ),
    r"time_embedding" : (
        "0",
        "2",
    ),
    r"time_projection" : (
        "1",
    ),
    
    r"blocks\.\d+" : (
        "self_attn.q",
        "self_attn.k",
        "self_attn.v",
        "self_attn.o",
        "cross_attn.q",
        "cross_attn.k",
        "cross_attn.v",
        "cross_attn.o",
        "ffn.0",
        "ffn.2"
    ),
}