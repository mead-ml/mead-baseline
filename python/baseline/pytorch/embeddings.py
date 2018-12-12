import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from baseline.utils import Offsets
from baseline.embeddings import register_embeddings
from baseline.pytorch.torchy import (
    pytorch_embedding,
    ParallelConv,
    pytorch_linear,
    SkipConnection,
    Highway,
    pytorch_lstm,
    BiRNNWrapper,
)


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
        inv_timescales = torch.exp(torch.arange(0, self.dsz, 2).float() * -log_timescale_increment)

        pe = torch.zeros(mxlen, self.dsz)
        position = torch.arange(0, mxlen).float().unsqueeze(1)
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


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(nn.Module, PyTorchEmbeddings):
    def __init__(self, name, **kwargs):
        super(CharLSTMEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        weights = kwargs.get('weights')
        if weights is None:
            self.embeddings = nn.Embedding(self.vsz, self.dsz, padding_idx=Offsets.PAD)
        else:
            self.embeddings = pytorch_embedding(weights)
        self.lstmsz = kwargs.get('lstmsz', 50)
        layers = kwargs.get('layers', 1)
        pdrop = kwargs.get('pdrop', 0.5)
        rnn_type = kwargs.get('rnn_type', 'blstm')
        unif = kwargs.get('unif', 0)
        weight_init = kwargs.get('weight_init', 'uniform')
        self.char_comp = BiRNNWrapper(pytorch_lstm(self.dsz, self.lstmsz, rnn_type, layers, pdrop, unif=unif, initializer=weight_init, batch_first=False), layers)


    def forward(self, xch):
        B, T, W = xch.shape
        flat_chars = xch.view(-1, W)
        char_embeds = self.embeddings(flat_chars)

        # Lengths
        lengths = torch.sum(flat_chars != Offsets.PAD, dim=1)
        sorted_word_lengths, perm_idx = lengths.sort(0, descending=True)
        # Hotfix for no char spaces.
        sorted_word_lengths.masked_fill_(sorted_word_lengths == 0, 1)
        sorted_feats = char_embeds[perm_idx].transpose(0, 1).contiguous()

        packed = torch.nn.utils.rnn.pack_padded_sequence(sorted_feats, sorted_word_lengths.tolist())
        _, hidden = self.char_comp(packed)
        hidden = tuple(h[-1, :, :] for h in hidden)
        results = tuple(h.scatter_(0, perm_idx.unsqueeze(-1).expand_as(h), h) for h in hidden)
        return results[0].reshape((B, T, -1))

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.vsz
