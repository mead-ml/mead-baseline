import torch

def tensor_shape(tensor):
    return tensor.size()

def long_0_tensor_alloc(dims, dtype):
    lt = long_tensor_alloc(dims, dtype)
    lt.zero_()
    return lt

def long_tensor_alloc(dims, dtype):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(*dims)

