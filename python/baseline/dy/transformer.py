import math
import numpy as np
import dynet as dy
from baseline.dy.dynety import transpose, Linear, dynet_activation, LayerNorm


def subsequent_mask(size):
    mask = np.tril(np.ones((size, size, 1))).astype(np.uint8)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask), dy.inputTensor(inv_mask)


def subsequent_mask_list(size):
    mask = np.tril(np.ones((size, size))).astype(np.uint8)
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


def last_dim_softmax(x, softmax=dy.softmax):
    """Dynet lets you pick the dim in a softmax but only 0 or 1 so we reshape."""
    shape, batchsz = x.dim()
    flat = np.prod(shape[:-1])
    last = shape[-1]
    x = dy.reshape(x, (flat, last), batch_size=batchsz)
    x = softmax(x, d=1)
    return dy.reshape(x, shape, batch_size=batchsz)


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    d_k = query.dim()[0][-1]

    scores = batch_matmul(query, transpose(key, 0, 1)) / math.sqrt(d_k)
    if mask is not None:
        scores = dy.cmult(scores, mask[0]) + (mask[1] * -1e9)

    weights = last_dim_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(weights, value)

def scaled_dot_product_attention_list(query, key, value, mask=None, dropout=None):
    d_k = query.dim()[0][-1]
    sqrt_d_k = math.sqrt(d_k)
    scores = [(x * dy.transpose(y)) / sqrt_d_k for x, y in zip(query, key)]
    if mask is not None:
        scores = [dy.cmult(score, mask[0]) + (mask[1] * -1e9) for score in scores]
    weights = [dy.softmax(score, d=1) for score in scores]
    if dropout is not None:
        weights = [dy.dropout(weight, dropout) for weight in weights]
    return [w * v for w, v in zip(weights, value)]


def dot_product_attention(query, key, value, mask=None, dropout=None):
    scores = batch_matmul(query, transpose(key, -2, -1))
    if mask is not None:
        scores = dy.cmult(scores, mask[0]) + (mask[1] * -1e9)

    weights = last_dim_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(weights, value)


def dot_product_attention_list(query, key, value, mask=None, dropout=None):
    scores = [(x * dy.transpose(y)) for x, y in zip(query, key)]
    if mask is not None:
        scores = [dy.cmult(score, mask[0]) + (mask[1] * -1e9) for score in scores]
    weights = [dy.softmax(score, d=1) for score in scores]
    if dropout is not None:
        weights = [dy.dropout(weight, dropout) for weight in weights]
    return [w * v for w, v in zip(weights, value)]


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
        """Input shape ((T, H), B)"""
        shape, batchsz = query.dim()
        query = [p_Q(q) for q in query]
        query = [dy.reshape(q, (1, h, d_k), batch_size=batchsz) for q in query]
        query = dy.concatenate(query)
        query = transpose(query, 0, 1)
        key = [p_K(k) for k in key]
        key = [dy.reshape(k, (1, h, d_k), batch_size=batchsz) for k in key]
        key = dy.concatenate(key)
        key = transpose(key, 0, 1)
        value = [p_V(v) for v in value]
        value = [dy.reshape(v, (1, h, d_k), batch_size=batchsz) for v in value]
        value = dy.concatenate(value)
        value = transpose(value, 0, 1)
        drop = dropout if train else None
        x = attn(query, key, value, mask=mask, dropout=drop)
        x = dy.transpose(x, [1, 0, 2])
        x = dy.reshape(x, (shape[0], h * d_k), batch_size=batchsz)
        x = [p_O(i) for i in x]
        x = dy.transpose(dy.concatenate_cols(x))
        return x

    return run


def FFN(d_model, pdrop, pc, activation_type='relu', d_ff=None, name='ffn'):
    pc = pc.add_subcollection(name=name)
    if d_ff is None:
        d_ff = 4 * d_model
    expand = Linear(d_ff, d_model, pc, name='expand')
    contract = Linear(d_model, d_ff, pc, name='squeeze')
    act = dynet_activation(activation_type)

    def forward(xs, train=False):
        """Input shape: ((T, H), B)"""
        xs = [act(expand(x)) for x in xs]
        if train:
            xs = [dy.dropout(x, pdrop) for x in xs]
        xs = [squeeze(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        return xs


def TransformerEncoder(num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-encoder'):
    pc = pc.add_subcollection(name=name)
    self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    ffn = FFN(d_model, pdrop, pc, activation_type=activation_type, d_ff=d_ff)
    ln1 = LayerNorm(d_model, pc)
    ln2 = LayerNorm(d_model, pc)

    def forward(xs, mask=None, train=False):
        xs = [ln1(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        y = self_attn(x, x, x, mask, train)
        y = dy.dropout(y, pdrop) if train else y
        xs = xs + y

        xs = [ln2(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        y = ffn(xs, train)
        y = dy.dropout(y, pdrop) if train else y
        xs = xs + y

        return xs


def TransformerEncoderStack(num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-encoder-stack'):
    pc = pc.add_subcollection(name=name)
    layers = [TransformerEncoder(num_heads, d_model, pdrop, pc, scale=scale, activation_type=activation_type, d_ff=None) for _ in range(layers)]
    norm = LayerNorm(d_model, pc)

    def forward(xs, mask=None, train=False):
        for layer in layers:
            xs = layer(x, mask, train)
        xs = [norm(x) for x in xs]
        return dy.transpose(dy.concatenate_cols(xs))


def TransformerDecoder(num_heads, d_model, pdrop, pc, scale=True, activation_type='relu', d_ff=None, name='transformer-decoder'):
    pc = pc.add_subcollection(name=name)
    self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, pc, scale=scale)
    ffn = FFN(d_model, pdrop, pc, activation_type=activation_type, d_ff=d_ff)
    ln1 = LayerNorm(d_model, pc)
    ln2 = LayerNorm(d_model, pc)
    ln3 = LayerNorm(d_model, pc)

    def forward(xs, memory, src_mask, tgt_mask):
        xs = [ln1(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        y = self_attn(x, x, x, tgt_mask, train)
        y = dy.dropout(y, pdrop) if train else y
        xs = xs + y

        xs = [ln2(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        y = src_attn(x, memory, memory, src_mask)
        y = dy.dropout(y, pdrop) if train else y
        xs = xs + y

        xs = [ln3(x) for x in xs]
        xs = dy.transpose(dy.concatenate_cols(xs))
        y = ffn(x, train)
        y = dy.dropout(y, pdrop) if train else y
        xs = xs + y

        return xs


def TransformerDecoderStack(num_heads, d_model, pdrop, pc, scale=True, layers=1, activation_type='relu', d_ff=None, name='transformer-decoder-stack'):
    pc = pc.add_subcollection(name=name)
    layers = [TransformerDecoder(num_heads, d_model, pdrop, pc, scale, activation_type, d_ff) for _ in range(layers)]
    norm = LayerNorm(d_model, pc)

    def forward(x, memory, src_mask, tgt_mask, train):
        for layer in layers:
            x = layer(x, memory, src_mask, tgt_mask, train)
        xs = [norm(x) for x in xs]
        return dy.transpose(dy.concatenate_cols(xs))
