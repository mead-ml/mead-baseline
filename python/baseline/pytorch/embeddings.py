import torch
import numpy as np
from baseline.utils import lookup_sentence, get_version
from baseline.utils import crf_mask as crf_m
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from baseline.pytorch.torchy import pytorch_embedding, ParallelConv, pytorch_linear, pytorch_activation


class PyTorchEmbeddings(object):

    def __init__(self):
        super(PyTorchEmbeddings).__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def encode(self, x):
        return self(x)


class PyTorchWordEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, model, **kwargs):
        super(PyTorchWordEmbeddings, self).__init__()
        self.finetune = kwargs.get('finetune', True)
        self.vsz = model.get_vsz()
        self.dsz = model.get_dsz()
        self.embeddings = pytorch_embedding(model, self.finetune)
        print(self)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def forward(self, x):
        return self.embeddings(x)


def pytorch_embeddings(in_embeddings_obj, DefaultType=PyTorchWordEmbeddings, **kwargs):
    if isinstance(in_embeddings_obj, PyTorchEmbeddings):
        return in_embeddings_obj
    else:
        return DefaultType(in_embeddings_obj, **kwargs)


class PyTorchCharConvEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, model, **kwargs):
        super(PyTorchCharConvEmbeddings, self).__init__()
        self.embeddings = pytorch_embedding(model)
        self.vsz = model.get_vsz()
        char_filtsz = kwargs.get('cfiltsz', [3])
        char_hsz = kwargs.get('wsz', 30)
        activation_type = kwargs.get('activation', 'tanh')
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = ParallelConv(model.get_dsz(), char_hsz, char_filtsz, activation_type, pdrop)
        wchsz = self.char_comp.outsz
        self.linear = pytorch_linear(wchsz, wchsz)
        self.activation = pytorch_activation(activation_type)
        print(self)

    def get_dsz(self):
        return self.char_comp.outsz

    def get_vsz(self):
        return self.vsz

    def forward(self, xch):

        # For starters we need to perform embeddings for each character
        # (TxB) x W -> (TxB) x W x D
        _0, _1, W = xch.shape
        char_embeds = self.embeddings(xch.view(-1, W))
        # (TxB) x D x W
        char_vecs = char_embeds.transpose(1, 2).contiguous()

        #        pytorch_activation(self.activation_type)
        mots = self.char_comp(char_vecs)
        skipped = self.activation(self.linear(mots)) + mots
        return skipped.view(_0, _1, self.char_comp.outsz)

