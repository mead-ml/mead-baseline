import torch
import numpy as np

def tensor_max(tensor):
    return tensor.max()

def tensor_shape(tensor):
    return tensor.size()

def long_0_tensor_alloc(dims, dtype=None):
    lt = long_tensor_alloc(dims)
    lt.zero_()
    return lt

def long_tensor_alloc(dims, dtype=None):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)
