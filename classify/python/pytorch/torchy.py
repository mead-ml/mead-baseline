import torch

# Adapters for main functions to use torch.Tensor instead of numpy
class TorchExamples(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def width(self):
        return self.x.size(1)

def long_0_tensor_alloc2(dims, dtype):
    lt = long_tensor_alloc2(dims, dtype)
    lt.zero_()
    return lt

def long_tensor_alloc2(dims, dtype):
    if type(dims) == int or len(dims) == 1:
        return torch.LongTensor(dims)
    return torch.LongTensor(dims[0], dims[1])

