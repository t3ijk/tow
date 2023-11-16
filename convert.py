import json



L = []
with open('test_dict_model copy.json', 'r') as f:
    json_obj = json.load(f)
    for key in json_obj:

        if key == 'shared.weight':
            L.append('shared_embedding.weight')
        
        if 'encoder.block.' in key:
            segs = key.split(".")
            block_id = segs[2]
            layer_id = segs[4]
            compo_name  = segs[5]
            weight_name  = segs[6]
            if compo_name == 'SelfAttention':
                if weight_name == 'q':
                    L.append(f"encoder.{block_id}.multi_head_attention.WQ.weight")
                if weight_name == 'k':
                    L.append(f"encoder.{block_id}.multi_head_attention.WK.weight")
                if weight_name == 'v':
                    L.append(f"encoder.{block_id}.multi_head_attention.WV.weight")
                if weight_name == 'o':
                    L.append(f"encoder.{block_id}.multi_head_attention.linear.weigh")
                if weight_name == 'relative_attention_bias':
                    L.append(f"encoder.{block_id}.multi_head_attention.input_positional_encoding.relative_attention_bias.weight")
            
            if compo_name == 'layer_norm':
                L.append(f"encoder.{block_id}.normal{int(layer_id)+1}.weight")


            if compo_name == 'DenseReluDense':
                if weight_name == 'wi_0':
                    L.append(f"encoder.{block_id}.feed_forward.wi_0.weight")
                if weight_name == 'wi_1':
                    L.append(f"encoder.{block_id}.feed_forward.wi_1.weight")
                if weight_name == 'wo':
                    L.append(f"encoder.{block_id}.feed_forward.wo.weight")

        if key == 'encoder.final_layer_norm.weight':
            L.append('encoder_final_layer_norm.weight')

        if 'decoder.block.' in key:
            segs = key.split(".")
            block_id = segs[2]
            layer_id = segs[4]
            compo_name  = segs[5]
            weight_name  = segs[6]
            if compo_name == 'SelfAttention':
                if weight_name == 'q':
                    L.append(f"decoder.{block_id}.masked_multi_head_attention.WQ.weight")
                if weight_name == 'k':
                    L.append(f"decoder.{block_id}.masked_multi_head_attention.WK.weight")
                if weight_name == 'v':
                    L.append(f"decoder.{block_id}.masked_multi_head_attention.WV.weight")
                if weight_name == 'o':
                    L.append(f"decoder.{block_id}.masked_multi_head_attention.linear.weigh")
                if weight_name == 'relative_attention_bias':
                    L.append(f"decoder.{block_id}.multi_head_attention.input_positional_encoding.relative_attention_bias.weight")

            if compo_name == 'EncDecAttention':
                if weight_name == 'q':
                    L.append(f"decoder.{block_id}.multi_head_attention.WQ.weight")
                if weight_name == 'k':
                    L.append(f"decoder.{block_id}.multi_head_attention.WK.weight")
                if weight_name == 'v':
                    L.append(f"decoder.{block_id}.multi_head_attention.WV.weight")
                if weight_name == 'o':
                    L.append(f"decoder.{block_id}.multi_head_attention.linear.weigh")
                if weight_name == 'relative_attention_bias':
                    L.append(f"decoder.{block_id}.multi_head_attention.input_positional_encoding.relative_attention_bias.weight")        
            
            if compo_name == 'layer_norm':
                L.append(f"decoder.{block_id}.normal{int(layer_id)+1}.weight")


            if compo_name == 'DenseReluDense':
                if weight_name == 'wi_0':
                    L.append(f"decoder.{block_id}.feed_forward.wi_0.weight")
                if weight_name == 'wi_1':
                    L.append(f"decoder.{block_id}.feed_forward.wi_1.weight")
                if weight_name == 'wo':
                    L.append(f"decoder.{block_id}.feed_forward.wo.weight")            



        if key == 'decoder.final_layer_norm.weight':
            L.append('decoder_final_layer_norm.weight')

        if key == 'lm_head.weight':
            L.append('linear.weight')            


with open('l.json', "w") as f:
    json.dump(L, f) 