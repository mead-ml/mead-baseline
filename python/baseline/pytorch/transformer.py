import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from baseline.pytorch.torchy import pytorch_linear, pytorch_activation
from baseline.pytorch.torchy import pytorch_clone_module


def subsequent_mask(size):
    """
    Creates a lower triangular mask to mask future

    :param size: Temporal length
    :return: A tensor of type `uint8` that is 1s along diagonals and below, zero  o.w
    """
    attn_shape = (1, 1, size, size)
    sub_mask = np.tril(np.ones(attn_shape)).astype('uint8')
    return torch.from_numpy(sub_mask)


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

    We apply the query to the keys to recieve our weights via softmax, which are then applied
    for each value, but in a series of efficient matrix operations.  In the case of self-attention,
    the key, query and values are all low order projections of the same input.

    :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
    :param key: a set of keys from encoder or self
    :param value: a set of values from encoder or self
    :param mask: masking (for destination) to prevent seeing what we shouldnt
    :param dropout: apply dropout operator post-attention (this is not a float)
    :return: A tensor that is (BxHxTxT)

    """
    # (., H, T, T) = (., H, T, D) x (., H, D, T)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


def dot_product_attention(query, key, value, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """
    def __init__(self, h, d_model, dropout=0.1, scale=False):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_Q = pytorch_linear(d_model, d_model)
        self.w_K = pytorch_linear(d_model, d_model)
        self.w_V = pytorch_linear(d_model, d_model)
        self.w_O = pytorch_linear(d_model, d_model)
        self.attn_fn = scaled_dot_product_attention if scale else dot_product_attention
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        batchsz = query.size(0)

        # (B, H, T, D)
        query = self.w_Q(query).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_K(key).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_V(value).view(batchsz, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attn_fn(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(batchsz, -1, self.h * self.d_k)
        return self.w_O(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, activation_type='relu', d_ff=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale)
        self.ffn = nn.Sequential(pytorch_linear(self.d_model, self.d_ff),
                                 pytorch_activation(activation_type),
                                 pytorch_linear(self.d_ff, self.d_model))
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, mask=None):
        """
        :param x:
        :param mask:
        :return:
        """
        # Builtin Attention mask
        x = self.ln1(x)
        h = self.self_attn(x, x, x, mask)
        x = x + self.dropout(h)

        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, activation_type='relu', d_ff=None):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.self_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop, scale=scale)
        self.src_attn = MultiHeadedAttention(num_heads, self.d_model, pdrop, scale=scale)
        self.ffn = nn.Sequential(pytorch_linear(self.d_model, self.d_ff),
                                 pytorch_activation(activation_type),
                                 pytorch_linear(self.d_ff, self.d_model))

        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ln3 = nn.LayerNorm(self.d_model, eps=1e-12)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x, memory, src_mask, tgt_mask):

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn(x, x, x, tgt_mask))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn(x, memory, memory, src_mask))

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x))
        return x


class TransformerEncoderStack(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, layers=1, activation_type='relu', d_ff=None):
        super(TransformerEncoderStack, self).__init__()
        single_layer = TransformerEncoder(num_heads, d_model, pdrop, scale, activation_type, d_ff)
        self.layers = pytorch_clone_module(single_layer, layers)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderStack(nn.Module):
    def __init__(self, num_heads, d_model, pdrop, scale=True, layers=1, activation_type='relu', d_ff=None):
        super(TransformerDecoderStack, self).__init__()
        single_layer = TransformerDecoder(num_heads, d_model, pdrop, scale, activation_type, d_ff)
        self.layers = pytorch_clone_module(single_layer, layers)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
