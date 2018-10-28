import math
import numpy as np
import dynet as dy
from baseline.dy.dynety import transpose, Linear, dynet_activation, LayerNorm


def subsequent_mask(size):
    mask = np.triu(np.ones((size, size))).astype(np.uint8)
    mask = np.expand_dims(mask, -1)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask), dy.inputTensor(inv_mask)


def batch_matmul_last(x, y):
    """This is a pain to do.

    The idea is to:
        1. roll the matrix dimensions (last 2) with dy.transpose
        2. Move the extra dimensions into the batch dim with dy.reshape
        3. Do a normal matrix mult
        4. Reshape and roll back with transposes

    Note; This doesn't support broadcasting like the pytorch version.
    """
    ndim = len(x.dim()[0])
    axis = [ndim - 2, ndim - 1] + list(range(0, ndim - 2))
    x = dy.transpose(x, axis)
    y = dy.transpose(y, axis)
    x_shape, batchsz = x.dim()
    x_mat = x_shape[:2]
    inners = x_shape[2:]
    to_batch = np.prod(inners)
    y_shape, _ = y.dim()
    y_mat = y_shape[:2]

    x = dy.reshape(x, x_mat, batch_size=to_batch * batchsz)
    y = dy.reshape(y, y_mat, batch_size=to_batch * batchsz)

    z = x * y
    z = dy.reshape(z, tuple([x_mat[0], y_mat[1]] + list(inners)), batch_size=batchsz)
    axis = list(range(2, ndim)) + [0, 1]
    z = dy.transpose(z, axis)

    return z


def batch_matmul(x, y):
    """Matmul between first two layers but the rest are ignored."""
    x_shape, batchsz = x.dim()
    x_mat = x_shape[:2]
    sames = x_shape[2:]
    fold = np.prod(sames)
    y_shape, _ = y.dim()
    y_mat = y_shape[:2]

    x = dy.reshape(x, x_mat, batch_size=fold*batchsz)
    y = dy.reshape(y, y_mat, batch_size=fold*batchsz)

    z = x * y
    z = dy.reshape(z, tuple([x_mat[0], y_mat[1]] + list(sames)), batch_size=batchsz)
    return z


def folded_softmax(x, softmax=dy.softmax):
    """Dynet only allows for softmax on matrices."""
    shape, batchsz = x.dim()
    first = shape[0]
    flat = np.prod(shape[1:])
    x = dy.reshape(x, (first, flat), batch_size=batchsz)
    x = softmax(x, d=0)
    return dy.reshape(x, shape, batch_size=batchsz)


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """Input Shape: ((D, T, H), B)"""
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
    scores = batch_matmul(query, transpose(key, -2, -1))
    if mask is not None:
        scores = dy.cmult(scores, mask[0]) + (mask[1] * -1e9)

    weights = last_dim_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(weights, value)


def MultiHeadedAttention(h, d_model, dropout, pc, scale=False, name='multi-headed-attention'):
    assert d_model % h == 0
    pc = pc.add_subcollection(name=name)
    d_k = d_model // h
    p_Q = Linear(d_model, d_model, pc, name="linear-q")
    p_K = Linear(d_model, d_model, pc, name="linear-k")
    p_V = Linear(d_model, d_model, pc, name="linear-v")
    p_O = Linear(d_model, d_model, pc, name="linear-o")
    attn = scaled_dot_product_attention if scale else dot_product_attention

    def run(query, key, value, mask=None, train=False):
        """Input shape ((H, T), B)"""
        shape, batchsz = query.dim()
        T = shape[1]
        query = p_Q(query)
        query = dy.reshape(query, (d_k, h, T), batch_size=batchsz)
        query = transpose(query, 1, 2)

        key = p_K(key)
        key = dy.reshape(query, (d_k, h, T), batch_size=batchsz)
        key = transpose(key, 1, 2)

        value = p_V(value)
        value = dy.reshape(value, (d_k, h, T), batch_size=batchsz)
        value = transpose(key, 1, 2)

        drop = dropout if train else None
        x = attn(query, key, value, mask=mask, dropout=drop)
        x = transpose(x, 0, 1)
        x = dy.reshape(x, (h * d_k, T), batch_size=batchsz)
        return p_O(x)

    return run


def FFN(d_model, pdrop, pc, activation_type='relu', d_ff=None, name='ffn'):
    pc = pc.add_subcollection(name=name)
    if d_ff is None:
        d_ff = 4 * d_model
    expand = Linear(d_ff, d_model, pc, name='expand')
    contract = Linear(d_model, d_ff, pc, name='squeeze')
    act = dynet_activation(activation_type)

    def forward(x, train=False):
        """Input shape: ((H, T), B)"""
        x = act(expand(x))
        x = dy.dropout(x, pdrop) if train else x
        x = squeeze(x)
        return x

    return forward


def TransformerEncoder(num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-encoder'):
    pc = pc.add_subcollection(name=name)
    self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    ffn = FFN(d_model, pdrop, pc, activation_type=activation_type, d_ff=d_ff)
    ln1 = LayerNorm(d_model, pc)
    ln2 = LayerNorm(d_model, pc)

    def forward(x, mask=None, train=False):
        """Input shape: ((H, T), B)"""
        x = ln1(x)
        y = self_attn(x, x, x, mask, train)
        y = dy.dropout(y, pdrop) if train else y
        x = x + y

        x = ln2(x)
        y = ffn(x, train)
        y = dy.dropout(y, pdrop) if train else y
        x = x + y

        return x

    return forward


def TransformerEncoderStack(num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-encoder-stack'):
    pc = pc.add_subcollection(name=name)
    layers = [TransformerEncoder(num_heads, d_model, pdrop, pc, scale=scale, activation_type=activation_type, d_ff=None) for _ in range(layers)]
    norm = LayerNorm(d_model, pc)

    def forward(x, mask=None, train=False):
        """Input shape: ((H, T), B)"""
        for layer in layers:
            x = layer(x, mask, train)
        x = norm(x)
        return x

    return forward


def TransformerDecoder(num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-decoder'):
    pc = pc.add_subcollection(name=name)
    self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    ffn = FFN(d_model, pdrop, pc, activation_type=activation_type, d_ff=d_ff)
    ln1 = LayerNorm(d_model, pc)
    ln2 = LayerNorm(d_model, pc)
    ln3 = LayerNorm(d_model, pc)

    def forward(x, memory, src_mask, tgt_mask):
        """Input shape: ((H, T), B)"""
        x = ln1(x)
        y = self_attn(x, x, x, tgt_mask, train)
        y = dy.dropout(y, pdrop) if train else y
        x = x + y

        x = ln2(x)
        y = src_attn(x, memory, memory, src_mask)
        y = dy.dropout(y, pdrop) if train else y
        x = x + y

        x = ln3(x)
        y = ffn(x, train)
        y = dy.dropout(y, pdrop) if train else y
        x = x + y

        return x

    return forward


def TransformerDecoderStack(num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-decoder-stack'):
    pc = pc.add_subcollection(name=name)
    layers = [TransformerDecoder(num_heads, d_model, pdrop, pc, scale, activation_type, d_ff) for _ in range(layers)]
    norm = LayerNorm(d_model, pc)

    def forward(x, memory, src_mask, tgt_mask, train):
        """Input shape: ((H, T), B)"""
        for layer in layers:
            x = layer(x, memory, src_mask, tgt_mask, train)
        x = norm(x)
        return x

    return forward
