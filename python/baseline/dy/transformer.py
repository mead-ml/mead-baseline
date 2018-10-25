import math
import numpy as np
import dynet as dy
from baseline.dy.dynety import FoldedLinear


def subsequent_mask(size):
    mask = np.tril(np.ones((1, size, size))).astype(np.uint8)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask), dy.inputTensor(inv_mask)


def batch_matmul(x, y):
    """This is a pain to do.

    The idea is to:
        1. roll the matrix dimensions (last 2) with dy.transpose
        2. Move the extra dimensions into the batch dim with dy.reshape
        3. Do a normal matrix mult
        4. Reshape and roll back with transposes

    Note; This doesn't support broadcasting like the pytorch version.
    """
    ndim = len(x.dim()[0])
    axis = [ndim - 2 , ndim - 1] + list(range(0, ndim - 2))
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

def swap_last(x):
    """Swap the last two dims like torch.transpose(x, -2, -1)."""
    ndims = len(x.dim()[0])
    axis = list(range(ndims - 2)) + [ndims - 1, ndims - 2]
    return dy.transpose(x, axis)


def last_dim_softmax(x, softmax=dy.softmax):
    """Dynet lets you pick the dim in a softmax but only 0 or 1 so we reshape."""
    shape, batchsz = x.dim()
    flat = np.prod(shape[:-1])
    last = shape[-1]
    x = dy.reshape(x, (flat, last), batch_size=batchsz)
    x = softmax(x, d=1)
    return dy.reshape(x, shape, batch_size=batchsz)


def scaled_dot_product_attention(query, key, value, masks=None, dropout=None):
    d_k = query.dim()[0][-1]

    scores = batch_matmul(query, swap_last(key)) / math.sqrt(d_k)
    if masks is not None:
        scores = dy.cmult(scores, masks[0]) + (masks[1] * -1e9)

    weights = last_dim_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(weights, value)


def dot_product_attention(query, key, value, masks=None, dropout=None):
    scores = batch_matmul(query, swap_last(key))
    if masks is not None:
        scores = dy.cmult(scores, masks[0]) + (masks[1] * -1e9)

    weights = last_dim_softmax(scores)

    if dropout is not None:
        weights = dy.dropout(weights, dropout)

    return batch_matmul(weights, value)


def MultiHeadedAttention(h, d_model, dropout, pc, scale=False, name='multi-headed-attention'):
    assert d_model % h == 0
    pc = pc.add_subcollection(name=name)
    d_k = d_model // h
    p_Q = FoldedLinear(d_model, d_model, pc, name="linear-q")
    p_K = FoldedLinear(d_model, d_model, pc, name="linear-k")
    p_V = FoldedLinear(d_model, d_model, pc, name="linear-v")
    p_O = FoldedLinear(d_model, d_model, pc, name="linear-o")
    attn = scaled_dot_product_attention if scale else dot_product_attention

    def run(query, key, value, mask=None, train=False):
        shape, batchsz = query.dim()
        query = p_Q(query)
        query = dy.reshape(query, (shape[0], h, d_k), batch_size=batchsz)
        query = dy.transpose(query, [1, 0, 2])
        key = p_K(key)
        key = dy.reshape(key, (shape[0], h, d_k), batch_size=batchsz)
        key = dy.transpose(key, [1, 0, 2])
        value = p_V(value)
        value = dy.reshape(value, (shape[0], h, d_k), batch_size=batchsz)
        value = dy.transpose(value, [1, 0, 2])
        drop = dropout if train else None
        x = attn(query, key, value, masks=mask, dropout=drop)
        x = dy.transpose(x, [1, 0, 2])
        x = dy.reshape(x, (shape[0], h * d_k), batch_size=batchsz)
        return p_O(x)

    return run
