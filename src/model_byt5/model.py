import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from src.model_byt5.utils import _relative_position_bucket
import math
from enum import Enum

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

class AttentionType(Enum):
    ENCODER_ATTENTION = 'encoder_multi_header_attention'
    DECODER_MASKED_ATTENTION = 'decoder_masked_attention'
    DECODER_CROSS_ATTENTION = 'decoder_cross_attention'

class MultiHeadAttention(nn.Module):
    def __init__(self, need_add_position_encoding=False, type=AttentionType.ENCODER_ATTENTION):
        super().__init__()
        # https://arxiv.org/pdf/1706.03762.pdf
        # Linear weight together for all heads
        self.WQ = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.num_heads * CONFIG_T5.d_kv, bias=False)
        self.WK = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.num_heads * CONFIG_T5.d_kv, bias=False)
        self.WV = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.num_heads * CONFIG_T5.d_kv, bias=False)

        # Linear back to d_model
        self.linear = nn.Linear(CONFIG_T5.num_heads * CONFIG_T5.d_kv, CONFIG_T5.d_model, bias=False)
        self.need_add_position_encoding = need_add_position_encoding
        if self.need_add_position_encoding:
            self.input_positional_encoding = PositionalEncoding()

        self.attentionType = type  

    def forward(self, kv_sequences, q_sequences, mask=None):
        batch = kv_sequences.shape[0]
        kv_length = kv_sequences.shape[1]
        q_length = q_sequences.shape[1]

        # hidden_states, (batch, sequence_length, d_model)
        # 1. map hidden_states to q_heads, -> (batch, sequence_length, num_heads * d_kv)
        # 2. split q_heads to q_head, -> (batch, sequence_length, num_heads, d_kv)
        # 3. permute -> (batch, num_heads, sequence_length, d_kv)
        self.q = self.WQ(q_sequences).reshape([batch, q_length, CONFIG_T5.num_heads, CONFIG_T5.d_kv]).transpose(1, 2)
        self.k = self.WK(kv_sequences).reshape([batch, kv_length, CONFIG_T5.num_heads, CONFIG_T5.d_kv]).transpose(1, 2)
        self.v = self.WV(kv_sequences).reshape([batch, kv_length, CONFIG_T5.num_heads, CONFIG_T5.d_kv]).transpose(1, 2)

        # dot product
        logits = torch.matmul(
            self.q, self.k.transpose(3, 2)
        )

        # ??? cross attention no need pos_bias
        pos_bias = None
        if self.attentionType == AttentionType.ENCODER_ATTENTION:
            # AttentionType.ENCODER_ATTENTION 
            pos_coding = MODEL_T5.encoder[0].multi_head_attention.input_positional_encoding
            pos_bias = pos_coding(logits.shape[2], logits.shape[3], bidirectional=True)

        if self.attentionType == AttentionType.DECODER_MASKED_ATTENTION:
            # index 0, AttentionType.DECODER_MASKED_ATTENTION 
            pos_coding = MODEL_T5.decoder[0].masked_multi_head_attention.input_positional_encoding
            pos_bias = pos_coding(logits.shape[2], logits.shape[3], bidirectional=False)    
        
        # first layer need_add_position_encoding
        # (batch, num_heads, query_sequencies_length, key_sequencies_length) + (1, num_heads, query_sequencies_length, key_sequencies_length)

        if pos_bias is not None:
            logits += pos_bias

        if mask is not None:
                logits = logits + mask
        # ??? no scaled, according to the original paper?
        # logits = logits / (1.0 / math.sqrt(CONFIG_T5.d_kv))

        # print('logits', logits.shape,
        #       torch.var_mean(logits))

        # (batch, num_heads, query_length, key_length)
        attention_weights = nn.functional.softmax(logits.float(), dim=-1).type_as(
            logits
        )
        # new values for the queries, (batch, num_heads, query_length, d_kv)
        v_output = torch.matmul(attention_weights, self.v)
        # concat heads, (batch, query_length, num_heads*d_kv)
        v_output = v_output.transpose(2, 1).reshape([batch, q_length, CONFIG_T5.d_kv * CONFIG_T5.num_heads])
        # project back to d_model, (batch, query_length, d_model)
        v_output = self.linear(v_output)
        # print('v_output', v_output.shape,
        #       torch.var_mean(v_output))
        return v_output

class LayerNormal(nn.Module):
    def __init__(self, hidden_size=CONFIG_T5.d_model, eps=CONFIG_T5.layer_norm_epsilon):
        super().__init__()

        # learnable per-element affine parameters initialized to ones
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        nn.LayerNorm

    def forward(self, hidden_states):
        # ref: transformers modeling_t5.py T5LayerNorm
        # T5 Norm is only on the d_model dimension
        # T5 Norm only scales with var and doesn't shift with mean
        # T5 Norm no bias
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # (batch, query_length, d_model)
        return self.weight * hidden_states

class NewGELUActivation(nn.Module):
    # https://github.com/huggingface/transformers/blob/b074461ef0f54ce37c5239d30ee960ece28d11ec/src/transformers/activations.py#L49
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class FeedForward(nn.Module):
    # ReLU

    # def __init__(self):
    #     super().__init__()
    #     self.wi = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.d_ff, bias=False)
    #     self.wo = nn.Linear(CONFIG_T5.d_ff, CONFIG_T5.d_model, bias=False)
    #     self.act = nn.ReLU()
    # def forward(self, hidden_states):
    #     hidden_states = self.wi(hidden_states)
    #     hidden_states = self.act(hidden_states)
    #     hidden_states = self.wo(hidden_states)
    #     return hidden_states

    # gated GeLU, ref: hf transformers   
    def __init__(self):
        super().__init__()
        self.wi_0 = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.d_ff, bias=False)
        self.wi_1 = nn.Linear(CONFIG_T5.d_model, CONFIG_T5.d_ff, bias=False)
        self.wo = nn.Linear(CONFIG_T5.d_ff, CONFIG_T5.d_model, bias=False)
        self.dropout = nn.Dropout(CONFIG_T5.dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    
class EncoderLayer(nn.Module):
    def __init__(self, layer_index):
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
        self.multi_head_attention = MultiHeadAttention(need_add_position_encoding=(layer_index == 0), type=AttentionType.ENCODER_ATTENTION)
        self.normal2 = LayerNormal()
        self.feed_forward = FeedForward()

        self.need_input_stack_dropout = False
        self.need_output_stack_dropout = False
        if layer_index == 0:
            self.need_input_stack_dropout = True
            self.input_stack_dropout = nn.Dropout(CONFIG_T5.dropout_rate)
        
        if layer_index == CONFIG_T5.num_layers:
            self.need_output_stack_dropout = True
            self.output_stack_dropout = nn.Dropout(CONFIG_T5.dropout_rate) 
        self.dropout1 = nn.Dropout(CONFIG_T5.dropout_rate)
        self.dropout2 = nn.Dropout(CONFIG_T5.dropout_rate)
        

    def forward(self, hidden_states):
        # main and residual hidden_states
        if self.need_input_stack_dropout:
            hidden_states = self.input_stack_dropout(hidden_states)
        
        # paper residual graph?
        # hidden_states = self.normal1(hidden_states)
        # residual = self.dropout1(hidden_states)
        # hidden_states = self.multi_head_attention(kv_sequences=hidden_states, q_sequences=hidden_states)
        # hidden_states = hidden_states + residual
        # hidden_states = self.normal2(hidden_states)
        # residual = self.dropout2(hidden_states)
        # hidden_states = self.feed_forward(hidden_states)
        # hidden_states = hidden_states + residual

        # hf
        hidden_states_normalized = self.normal1(hidden_states)
        attention_outs = self.multi_head_attention(
            kv_sequences=hidden_states_normalized,
            q_sequences=hidden_states_normalized)
        hidden_states = hidden_states + self.dropout1(attention_outs)
        hidden_states_normalized = self.normal2(hidden_states)
        feed_forward_outs = self.feed_forward(hidden_states_normalized)
        hidden_states = hidden_states + self.dropout2(feed_forward_outs)
        if self.need_output_stack_dropout:
            hidden_states = self.output_stack_dropout(hidden_states)
       
        return hidden_states
    
class DecoderLayer(nn.Module):
    def __init__(self, layer_index):
        super().__init__()
        self.normal1 = LayerNormal()
        self.masked_multi_head_attention = MultiHeadAttention(need_add_position_encoding=(layer_index == 0), type=AttentionType.DECODER_MASKED_ATTENTION)
        self.normal2 = LayerNormal()
        # encoder-decoder attention 
        self.multi_head_attention = MultiHeadAttention(type=AttentionType.DECODER_CROSS_ATTENTION)
        self.normal3 = LayerNormal()
        self.feed_forward = FeedForward()
    
        self.need_input_stack_dropout = False
        self.need_output_stack_dropout = False
        if layer_index == 0:
            self.need_input_stack_dropout = True
            self.input_stack_dropout = nn.Dropout(CONFIG_T5.dropout_rate)
        
        if layer_index == CONFIG_T5.num_decoder_layers:
            self.need_output_stack_dropout = True
            self.output_stack_dropout = nn.Dropout(CONFIG_T5.dropout_rate)
        
        self.dropout1 = nn.Dropout(CONFIG_T5.dropout_rate)
        self.dropout2 = nn.Dropout(CONFIG_T5.dropout_rate)
        self.dropout3 = nn.Dropout(CONFIG_T5.dropout_rate)

    def forward(self, hidden_states, encoder_outs):
        # main and residual hidden_states
        if self.need_input_stack_dropout:
            hidden_states = self.input_stack_dropout(hidden_states)
        
        # ??? different skip-residual-dropout graph to the t5 paper?
        hidden_states_normalized = self.normal1(hidden_states)
        attention_outs = self.masked_multi_head_attention(kv_sequences=hidden_states_normalized, q_sequences=hidden_states_normalized, mask=MODEL_T5.mask_for_masked_attention)
        hidden_states = hidden_states + self.dropout1(attention_outs)
        hidden_states_normalized = self.normal2(hidden_states)
        attention_outs = self.multi_head_attention(kv_sequences=encoder_outs, q_sequences=hidden_states_normalized)
        hidden_states = hidden_states + self.dropout2(attention_outs)
        hidden_states_normalized = self.normal3(hidden_states)
        feed_forward_outs = self.feed_forward(hidden_states_normalized)
        hidden_states = hidden_states + self.dropout3(feed_forward_outs)
        
        if self.need_output_stack_dropout:
            hidden_states = self.output_stack_dropout(hidden_states)
        return hidden_states

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.relative_attention_bias = nn.Embedding(
            CONFIG_T5.relative_attention_num_buckets, CONFIG_T5.num_heads
        )

    def forward(self, query_length, key_length, device=None, bidirectional=True):
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
        return values

class Transformer_byt5(nn.Module):
    def __init__(self, config_={}):
        super().__init__()
        CONFIG_T5 = Config_byt5(**config_)
        self.shared_embedding = nn.Embedding(
            CONFIG_T5.vocab_size, CONFIG_T5.d_model)
        self.encoder = nn.ModuleList([EncoderLayer(i) for i in range(CONFIG_T5.num_layers)])
        self.encoder_final_layer_norm = LayerNormal()
        self.decoder = nn.ModuleList([DecoderLayer(i) for i in range(CONFIG_T5.num_decoder_layers)])
        self.decoder_final_layer_norm = LayerNormal()
        self.linear = nn.Linear(
            CONFIG_T5.d_model, CONFIG_T5.vocab_size, bias=False)

        global MODEL_T5
        MODEL_T5 = self
        MODEL_T5.mask_for_masked_attention = None

        self.cached_decoder_final_outputs = None
        self.cached_encoder_final_hidden_states = None

    def get_attention_mask(self, seq_length):
        seq_ids = torch.arange(seq_length)
        causal_mask = seq_ids[None, :].repeat(1, seq_length, 1) <= seq_ids[None, :, None]
        causal_mask = causal_mask.to(torch.float32).unsqueeze(0)
        causal_mask = (1.0 - causal_mask) * torch.finfo(torch.float32).min
        return causal_mask
        
    def generate(self, inputs, max_length):
        return self.forward(inputs)
        
    def forward(self, inputs, labels=None, last_outputs=None):
        # encode
        if self.cached_encoder_final_hidden_states is None:
            encoder_hidden_states = self.shared_embedding(inputs)
            for i, layer in enumerate(self.encoder):
                encoder_hidden_states = layer(encoder_hidden_states)
            
            encoder_hidden_states = self.encoder_final_layer_norm(encoder_hidden_states)
            self.cached_encoder_final_hidden_states = encoder_hidden_states

        # prepare inputs for decoder
        last_outputs = self.cached_decoder_final_outputs
        if last_outputs is None:
            last_outputs = torch.zeros(inputs.shape[0], 0)
        if labels is not None:
            last_outputs = labels[:, :-1]
        
        # insert [0] and shift right
        shifted_input_ids = last_outputs.new_zeros(last_outputs.shape[0], last_outputs.shape[1] + 1)
        shifted_input_ids[:, 1:] = last_outputs.clone()
        shifted_input_ids[:, 0] = 0            

        # decode
        MODEL_T5.mask_for_masked_attention = self.get_attention_mask(shifted_input_ids.shape[1])
        decoder_hidden_states = self.shared_embedding(shifted_input_ids)
        for i, layer in enumerate(self.decoder):
            decoder_hidden_states = layer(decoder_hidden_states, self.cached_encoder_final_hidden_states)
        decoder_hidden_states = self.decoder_final_layer_norm(decoder_hidden_states)
        output_logits = self.linear(decoder_hidden_states)

        # calculate loss
        if labels is not None:
            predicts = output_logits.view(-1, output_logits.size(-1))
            labels = labels.view(-1)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(predicts, labels)

        return {
            "output_logits": output_logits,
            "loss": loss
        }
