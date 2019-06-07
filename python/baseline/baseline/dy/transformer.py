import math
import numpy as np
import dynet as dy
from baseline.dy.dynety import (
    DynetLayer,
    Linear, LayerNorm,
    transpose, unsqueeze,
    dynet_activation, batch_matmul, folded_softmax
)


def subsequent_mask(T):
    """Build a mask to hide the future from self attention.

    Output: ((T, T, 1), 1) to broadcast over both the heads and the batch

    Returns:
        (dy.Expression, dy.Expression)
        - The first mask has ones in valid positions and zeros at invalid. This
          is used to zero out the future using a `dy.cmult`.
        - The second mask has 1 at invalid positions and zeros at valid. This
          can be used to fill invalid positions with negative numbers via
          addition.
    """
    mask = np.triu(np.ones((T, T))).astype(np.uint8)
    mask = np.expand_dims(mask, -1)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask), dy.inputTensor(inv_mask)


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Input Shape: ((D, T, H), B) Output: ((D, T, H), B)"""
    d_k = query.dim()[0][0]

    scores = batch_matmul(transpose(key, 0, 1), query) / math.sqrt(d_k)
    if mask is not None:
        scores = dy.cmult(scores, mask[0]) + (mask[1] * -1e9)

    weights = folded_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(value, weights)


def dot_product_attention(query, key, value, mask=None, dropout=None):
    """Input Shape: ((D, T, H), B)"""
    scores = batch_matmul(transpose(key, 0, 1), query)
    if mask is not None:
        scores = dy.cmult(scores, mask[0]) + (mask[1] * -1e9)

    weights = folded_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(value, weights)


class MultiHeadedAttention(DynetLayer):
    def __init__(self, h, d_model, dropout, pc, scale=False, name='multi-headed-attention'):
        assert d_model % h == 0
        pc = pc.add_subcollection(name=name)
        super(MultiHeadedAttention, self).__init__(pc)
        self.d_k = d_model // h
        self.h = h
        self.p_Q = Linear(d_model, d_model, pc, name="linear-q")
        self.p_K = Linear(d_model, d_model, pc, name="linear-k")
        self.p_V = Linear(d_model, d_model, pc, name="linear-v")
        self.p_O = Linear(d_model, d_model, pc, name="linear-o")
        self.attn = scaled_dot_product_attention if scale else dot_product_attention
        self.pdrop = dropout

    def __call__(self, query, key, value, mask=None, train=False):
        """Input: ((H, T), B) Output: ((H, T), B)"""
        _, batchsz = query.dim()
        query = self.p_Q(query)
        t = query.dim()[0][1]
        query = dy.reshape(query, (self.d_k, self.h, t), batch_size=batchsz)
        query = transpose(query, 1, 2)

        key = self.p_K(key)
        t = key.dim()[0][1]
        key = dy.reshape(key, (self.d_k, self.h, t), batch_size=batchsz)
        key = transpose(key, 1, 2)

        value = self.p_V(value)
        t = value.dim()[0][1]
        value = dy.reshape(value, (self.d_k, self.h, t), batch_size=batchsz)
        value = transpose(value, 1, 2)

        pdrop = self.pdrop if train else None
        x = self.attn(query, key, value, mask=mask, dropout=pdrop)
        x = transpose(x, 1, 2)
        t = x.dim()[0][2]
        x = dy.reshape(x, (self.h * self.d_k, t), batch_size=batchsz)
        return self.p_O(x)


class FFN(DynetLayer):
    def __init__(self, d_model, pdrop, pc, activation_type='relu', d_ff=None, name='ffn'):
        pc = pc.add_subcollection(name=name)
        super(FFN, self).__init__(pc)
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.expand = Linear(d_ff, d_model, self.pc, name='expand')
        self.contract = Linear(d_model, d_ff, self.pc, name='contract')
        self.act = dynet_activation(activation_type)
        self.pdrop = pdrop

    def __call__(self, x, train=False):
        """Input: ((H, T), B) Output: ((H, T), B)."""
        x = self.act(self.expand(x))
        x = dy.dropout(x, self.pdrop) if train else x
        return self.contract(x)


class TransformerEncoder(DynetLayer):
    def __init__(self, num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-encoder'):
        pc = pc.add_subcollection(name=name)
        super(TransformerEncoder, self).__init__(pc)
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, self.pc, scale=scale)
        self.ffn = FFN(d_model, pdrop, self.pc, activation_type=activation_type, d_ff=d_ff)
        self.ln1 = LayerNorm(d_model, self.pc)
        self.ln2 = LayerNorm(d_model, self.pc)
        self.pdrop = pdrop

    def __call__(self, x, mask=None, train=False):
        """Input: ((H, T), B)"""
        x = self.ln1(x)
        y = self.self_attn(x, x, x, mask, train)
        y = dy.dropout(y, self.pdrop) if train else y
        x = x + y

        x = self.ln2(x)
        y = self.ffn(x, train)
        y = dy.dropout(y, self.pdrop) if train else y
        x = x + y

        return x


class TransformerEncoderStack(DynetLayer):
    def __init__(self, num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-encoder-stack'):
        pc = pc.add_subcollection(name=name)
        super(TransformerEncoderStack, self).__init__(pc)
        self.layers = [TransformerEncoder(num_heads, d_model, pdrop, self.pc, scale=scale, activation_type=activation_type, d_ff=None) for _ in range(layers)]
        self.norm = LayerNorm(d_model, pc)
        self.pdrop = pdrop

    def __call__(self, x, mask=None, train=False):
        for layer in self.layers:
            x = layer(x, mask, train)
        return self.norm(x)


class TransformerDecoder(DynetLayer):
    def __init__(self, num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-decoder'):
        pc = pc.add_subcollection(name=name)
        super(TransformerDecoder, self).__init__(pc)
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, self.pc, scale=scale)
        self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, self.pc, scale=scale)
        self.ffn = FFN(d_model, pdrop, self.pc, activation_type=activation_type, d_ff=d_ff)
        self.ln1 = LayerNorm(d_model, self.pc)
        self.ln2 = LayerNorm(d_model, self.pc)
        self.ln3 = LayerNorm(d_model, self.pc)
        self.pdrop = pdrop

    def __call__(self, x, memory, src_mask, tgt_mask, train=False):
        """Input shape: ((H, T), B)"""
        x = self.ln1(x)
        y = self.self_attn(x, x, x, tgt_mask, train)
        y = dy.dropout(y, self.pdrop) if train else y
        x = x + y

        x = self.ln2(x)
        y = self.src_attn(x, memory, memory, src_mask)
        y = dy.dropout(y, self.pdrop) if train else y
        x = x + y

        x = self.ln3(x)
        y = self.ffn(x, train)
        y = dy.dropout(y, self.pdrop) if train else y
        x = x + y

        return x


class TransformerDecoderStack(DynetLayer):
    def __init__(self, num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-decoder-stack'):
        pc = pc.add_subcollection(name=name)
        super(TransformerDecoderStack, self).__init__(pc)
        self.layers = [TransformerDecoder(num_heads, d_model, pdrop, self.pc, scale=scale, activation_type=activation_type, d_ff=d_ff) for _ in range(layers)]
        self.norm = LayerNorm(d_model, self.pc)

    def __call__(self, x, memory, src_mask, tgt_mask, train=False):
        """Input: ((H, T), B)"""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, train)
        return self.norm(x)
