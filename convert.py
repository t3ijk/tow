import json


def hf_model_weight_name_mapping(key):
    if key == 'shared.weight':
        return 'shared_embedding.weight'
    
    if 'encoder.block.' in key:
        segs = key.split(".")
        block_id = segs[2]
        layer_id = segs[4]
        compo_name  = segs[5]
        weight_name  = segs[6]
        if compo_name == 'SelfAttention':
            if weight_name == 'q':
                return f"encoder.{block_id}.multi_head_attention.WQ.weight"
            if weight_name == 'k':
                return f"encoder.{block_id}.multi_head_attention.WK.weight"
            if weight_name == 'v':
                return f"encoder.{block_id}.multi_head_attention.WV.weight"
            if weight_name == 'o':
                return f"encoder.{block_id}.multi_head_attention.linear.weight"
            if weight_name == 'relative_attention_bias':
                return f"encoder.{block_id}.multi_head_attention.input_positional_encoding.relative_attention_bias.weight"
        
        if compo_name == 'layer_norm':
            return f"encoder.{block_id}.normal{int(layer_id)+1}.weight"


        if compo_name == 'DenseReluDense':
            if weight_name == 'wi_0':
                return f"encoder.{block_id}.feed_forward.wi_0.weight"
            if weight_name == 'wi_1':
                return f"encoder.{block_id}.feed_forward.wi_1.weight"
            if weight_name == 'wo':
                return f"encoder.{block_id}.feed_forward.wo.weight"

    if key == 'encoder.final_layer_norm.weight':
        return 'encoder_final_layer_norm.weight'

    if 'decoder.block.' in key:
        segs = key.split(".")
        block_id = segs[2]
        layer_id = segs[4]
        compo_name  = segs[5]
        weight_name  = segs[6]
        if compo_name == 'SelfAttention':
            if weight_name == 'q':
                return f"decoder.{block_id}.masked_multi_head_attention.WQ.weight"
            if weight_name == 'k':
                return f"decoder.{block_id}.masked_multi_head_attention.WK.weight"
            if weight_name == 'v':
                return f"decoder.{block_id}.masked_multi_head_attention.WV.weight"
            if weight_name == 'o':
                return f"decoder.{block_id}.masked_multi_head_attention.linear.weight"
            if weight_name == 'relative_attention_bias':
                return f"decoder.{block_id}.masked_multi_head_attention.input_positional_encoding.relative_attention_bias.weight"

        if compo_name == 'EncDecAttention':
            if weight_name == 'q':
                return f"decoder.{block_id}.multi_head_attention.WQ.weight"
            if weight_name == 'k':
                return f"decoder.{block_id}.multi_head_attention.WK.weight"
            if weight_name == 'v':
                return f"decoder.{block_id}.multi_head_attention.WV.weight"
            if weight_name == 'o':
                return f"decoder.{block_id}.multi_head_attention.linear.weight"
            if weight_name == 'relative_attention_bias':
                return f"decoder.{block_id}.multi_head_attention.input_positional_encoding.relative_attention_bias.weight"        
        
        if compo_name == 'layer_norm':
            return f"decoder.{block_id}.normal{int(layer_id)+1}.weight"

        if compo_name == 'DenseReluDense':
            if weight_name == 'wi_0':
                return f"decoder.{block_id}.feed_forward.wi_0.weight"
            if weight_name == 'wi_1':
                return f"decoder.{block_id}.feed_forward.wi_1.weight"
            if weight_name == 'wo':
                return f"decoder.{block_id}.feed_forward.wo.weight"            

    if key == 'decoder.final_layer_norm.weight':
        return 'decoder_final_layer_norm.weight'

    if key == 'lm_head.weight':
        return 'linear.weight'        

# L = {}

# with open('temp/test_dict_model copy.json', 'r') as f:
#     json_obj = json.load(f)
#     for key in json_obj:
#         L[key] = hf_model_weight_name_mapping(key)



# with open('l.json', "w") as f:
#     json.dump(L, f) 