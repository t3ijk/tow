import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class Config_byt5:
        d_ff: int = 3584
        d_kv: int = 64
        d_model: int = 1472
        decoder_start_token_id: int = 0
        dropout_rate: float = 0.1
        eos_token_id: int = 1
        feed_forward_proj: str = "gated-gelu"
        gradient_checkpointing: bool = False
        initializer_factor: float = 1.0
        is_encoder_decoder: bool = True
        layer_norm_epsilon: float = 1e-06
        num_decoder_layers: int = 4
        num_heads: int = 6
        num_layers: int = 12
        pad_token_id: int = 0
        relative_attention_num_buckets: int = 32
        tie_word_embeddings: bool = False
        use_cache: bool =  True
        vocab_size: int = 384

class MultiHeadAttention(nn.Module):
        def __init__(self):
             super().__init__()


class MaskedMultiHeadAttention(nn.Module):
        def __init__(self):
             super().__init__()             

class AddAndNormal(nn.Module):
        def __init__(self):
             super().__init__()

class FeedForward(nn.Module):
        def __init__(self):
             super().__init__()

class EncoderLayer(nn.Module):
     def __init__(self):
        super().__init__()
        self.multi_head_attention  = MultiHeadAttention()
        self.add_and_normal1  = AddAndNormal()
        self.feed_forward  = FeedForward()
        self.add_and_normal2  = AddAndNormal()


class DecoderLayer(nn.Module):
     def __init__(self):
        super().__init__()
        self.masked_multi_head_attention  = MaskedMultiHeadAttention()
        self.add_and_normal1  = AddAndNormal()
        self.multi_head_attention  = MultiHeadAttention()
        self.add_and_normal2  = AddAndNormal()
        self.feed_forward  = FeedForward()
        self.add_and_normal3  = AddAndNormal()

class PositionalEncoding(nn.Module):
     def __init__(self):
        super().__init__()        

class Transformer_byt5 (nn.Module):
    def __init__(self, config_={}):
        super().__init__()

        config = Config_byt5(**config_)
        # ??? Linear == Embedding? https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518/9
        # self.input_embedding = nn.Linear(config.vocab_size, config.d_model, bias = False)
        self.input_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_positional_encoding = PositionalEncoding()
        self.encoder = nn.ModuleList([EncoderLayer() for i in range(config.num_layers)])

        self.output_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.output_positional_encoding = PositionalEncoding()
        self.decoder = nn.ModuleList([DecoderLayer() for i in range(config.num_decoder_layers)])

        self.linear = nn.Linear(config.d_model, config.vocab_size, bias = False)
        self.softmax = nn.Softmax()




# LayerNorm(x + Sublayer(x)) https://arxiv.org/pdf/1706.03762.pdf
# "Removing the Layer Norm bias, placing the layer normalization outside the residual path" (https://arxiv.org/pdf/1910.10683.pdf)

# https://github.com/google/seqio/blob/ad8e49e3b7fb6cee15f5a63a9e016bb63cc4f2e2/seqio/vocabularies.py#L518

# LARGEST_CODEPOINT = 0x10FFFF  # Decimal: 1,114,111 （UTF-8 Encoding:	0xF4 0x8F 0xBF 0xBF）so 244，so  12 个 