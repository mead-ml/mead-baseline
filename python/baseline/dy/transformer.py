import numpy as np
import dynet as dy


def subsequent_mask(size):
    mask = np.tril(np.ones((1, size, size))).astype(np.uint8)
    inv_mask = (mask == 0).astype(np.uint8)
    return dy.inputTensor(mask), dy.inputTensor(inv_mask)


def scaled_dot_product_attention(query, key, value, masks=None, dropout=None):
    d_q = query.dim()[0][-1]
    dims = list(range(len(query.dim()[0])))
    dims[-1] = dims[-2]
    dims[-2] = len(dims) - 1
    q_shape = query.dim()[0]
    q_first = np.prod(q_shape[:-1])

    key_t = dy.transpose(key, dims=dims)
    k_shape = key_t.dim()[0]
    k_first = np.prod(k_shape[:-1])
    d_k = k_shape[-1]

    print(query.dim())
    query = dy.reshape(query, (q_first, d_q))
    print(query.dim())
    print(key_t.dim())
    key_t = dy.reshape(key_t, (k_first, d_k))
    print(key_t.dim())

    scores = query * dy.transpose(key_t) / math.sqrt(d_q)
    print(scores.dim())
    if masks is not None:
        scores = dy.cmult(scores, masks[0]) + (masks[1] * -1e9)
    weights = dy.softmax(scores, d=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights * value, weights
