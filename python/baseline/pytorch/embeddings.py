from baseline.embeddings import register_embeddings
import torch.nn as nn
from collections import OrderedDict
from baseline.pytorch.torchy import (pytorch_embedding,
                                     ParallelConv,
                                     pytorch_linear,
                                     SkipConnection,
                                     Highway)

import numpy as np


class PyTorchEmbeddings(object):

    def __init__(self):
        super(PyTorchEmbeddings).__init__()

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def encode(self, x):
        return self(x)

    @classmethod
    def create(cls, model, name, **kwargs):
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


@register_embeddings(name='default')
class LookupTableEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, _, **kwargs):
        super(LookupTableEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        weights = kwargs.get('weights')
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=0)
        else:
            self.embeddings = pytorch_embedding(weights, self.finetune)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def forward(self, x):
        return self.embeddings(x)



@register_embeddings(name='char-conv')
class CharConvEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, _, **kwargs):
        super(CharConvEmbeddings, self).__init__()

        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        self.params = kwargs
        self.wsz = None
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        weights = kwargs.get('weights')
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=0)
        else:
            self.embeddings = pytorch_embedding(weights)
        char_filtsz = kwargs.get('cfiltsz', [3])
        char_hsz = kwargs.get('wsz', 30)
        activation_type = kwargs.get('activation', 'tanh')
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = ParallelConv(self.dsz, char_hsz, char_filtsz, activation_type, pdrop)
        wchsz = self.char_comp.outsz
        self.linear = pytorch_linear(wchsz, wchsz)
        gating = kwargs.get('gating', 'skip')
        GatingConnection = SkipConnection if gating == 'skip' else Highway
        num_gates = kwargs.get('num_gates', 1)
        self.gating_seq = nn.Sequential(OrderedDict(
            [('gate-{}'.format(i), GatingConnection(wchsz)) for i in range(num_gates)]
        ))
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
        gated = self.gating_seq(mots)
        return gated.view(_0, _1, self.char_comp.outsz)
