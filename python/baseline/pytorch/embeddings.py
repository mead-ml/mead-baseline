from baseline.embeddings import register_embeddings
import torch.nn as nn
from collections import OrderedDict
from baseline.pytorch.torchy import (pytorch_embedding,
                                     ParallelConv,
                                     pytorch_linear,
                                     SkipConnection,
                                     Highway)
import torch
import numpy as np
import math


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
        x = x.to(torch.long)
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
        xch = xch.to(torch.long)
        char_embeds = self.embeddings(xch.view(-1, W))
        # (TxB) x D x W
        char_vecs = char_embeds.transpose(1, 2).contiguous()

        #        pytorch_activation(self.activation_type)
        mots = self.char_comp(char_vecs)
        gated = self.gating_seq(mots)
        return gated.view(_0, _1, self.char_comp.outsz)


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, _, **kwargs):
        super(PositionalLookupTableEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        self.finetune = kwargs.get('finetune', True)
        # This could get us in trouble, if in doubt, pick something big
        mxlen = kwargs.get('mxlen', 1000)
        max_timescale = kwargs.get('max_timescale', 1.0e4)

        weights = kwargs.get('weights')
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=0)
        else:
            self.embeddings = pytorch_embedding(weights, self.finetune)

        log_timescale_increment = math.log(max_timescale) / self.dsz
        inv_timescales = torch.exp(torch.arange(0, self.dsz, 2) * -log_timescale_increment)

        pe = torch.zeros(mxlen, self.dsz)
        position = torch.arange(0, mxlen).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * inv_timescales)
        pe[:, 1::2] = torch.cos(position * inv_timescales)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def forward(self, x):
        """Add a positional encoding to the embedding, followed by dropout

        :param x: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        x = self.embeddings(x) * math.sqrt(self.dsz)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
