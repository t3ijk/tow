import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from model_byt5.utils import _relative_position_bucket

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
    relative_attention_max_distance: int = 128
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 384

CONFIG_T5: Config_byt5 = Config_byt5()

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()


class LayerNormal(nn.Module):
    def __init__(self, hidden_size=CONFIG_T5.d_model, eps=CONFIG_T5.layer_norm_epsilon):
        super().__init__()

        # learnable per-element affine parameters initialized to ones
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        nn.LayerNorm

    def forward(self, hidden_states):
        # ref: transformers modeling_t5.py T5LayerNorm
        # T5 Norm is only on last dimension https://stats.stackexchange.com/questions/620002/why-is-the-layer-normalization-same-with-the-instance-normalization-in-transform
        # T5 Norm only scales with var and doesn't shift with mean
        # T5 Norm no bias
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
       

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()


class EncoderLayer(nn.Module):
    def __init__(self):
        """
        paper: https://arxiv.org/pdf/1910.10683.pdf
        The encoder consists
        of a stack of “blocks”, each of which comprises two subcomponents: a self-attention layer
        followed by a small feed-forward network. Layer normalization (Ba et al., 2016) is applied to
        the input of each subcomponent. We use a simplified version of layer normalization where
        the activations are only rescaled and no additive bias is applied. After layer normalization,
        a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
        Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip
        connection, on the attention weights, and at the input and output of the entire stack  
        """
        super().__init__()
        self.normal1 = LayerNormal()
        # skip residual
        self.multi_head_attention = MultiHeadAttention()
        # add residual
        self.normal2 = LayerNormal()
        # skip residual
        self.feed_forward = FeedForward()
        # add residual


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_multi_head_attention = MaskedMultiHeadAttention()
        self.add_and_normal1 = LayerNormal()
        self.multi_head_attention = MultiHeadAttention()
        self.add_and_normal2 = LayerNormal()
        self.feed_forward = FeedForward()
        self.add_and_normal3 = LayerNormal()


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.relative_attention_bias = nn.Embedding(
            CONFIG_T5.relative_attention_num_buckets, CONFIG_T5.num_heads
        )

    def forward(self, query_length, key_length, device=None, bidirectional=True):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = _relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=bidirectional,
            num_buckets=CONFIG_T5.relative_attention_num_buckets,
            max_distance=CONFIG_T5.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)

        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        print(values, values.shape)
        return values


class Transformer_byt5(nn.Module):
    def __init__(self, config_={}):
        super().__init__()

        CONFIG_T5 = Config_byt5(**config_)
        # Linear == Embedding? https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518/9
        # self.input_embedding = nn.Linear(config.vocab_size, config.d_model, bias = False)
        self.input_embedding = nn.Embedding(
            CONFIG_T5.vocab_size, CONFIG_T5.d_model)
        self.input_positional_encoding = PositionalEncoding()
        self.encoder = nn.ModuleList(
            [EncoderLayer() for i in range(CONFIG_T5.num_layers)]
        )

        self.output_embedding = nn.Embedding(
            CONFIG_T5.vocab_size, CONFIG_T5.d_model)
        self.output_positional_encoding = PositionalEncoding()
        self.decoder = nn.ModuleList(
            [DecoderLayer() for i in range(CONFIG_T5.num_decoder_layers)]
        )

        self.linear = nn.Linear(
            CONFIG_T5.d_model, CONFIG_T5.vocab_size, bias=False)
        self.softmax = nn.Softmax()
