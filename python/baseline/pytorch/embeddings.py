import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from baseline.utils import Offsets, is_sequence
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


class PyTorchEmbeddings(nn.Module):

    def __init__(self, _=None, **kwargs):
        super(PyTorchEmbeddings, self).__init__()

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
class LookupTableEmbeddings(PyTorchEmbeddings):

    def __init__(self, _, **kwargs):
        super(LookupTableEmbeddings, self).__init__(_, **kwargs)
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
class CharConvEmbeddings(PyTorchEmbeddings):

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
        if is_sequence(char_filtsz[0]):
            char_hsz = [pair[1] for pair in char_filtsz]
            char_filtsz = [pair[0] for pair in char_filtsz]
        else:
            char_hsz = kwargs.get('wsz', 30)

        activation_type = kwargs.get('activation', 'tanh')
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = ParallelConv(self.dsz, char_hsz, char_filtsz, activation_type, pdrop)
        wchsz = self.char_comp.outsz
        self.linear = pytorch_linear(wchsz, wchsz)
        gating = kwargs.get('gating', 'skip')
        GatingConnection = SkipConnection if gating == 'skip' else Highway
        num_gates = kwargs.get('num_gates', 1)

        gates = [('gate-{}'.format(i), GatingConnection(wchsz)) for i in range(num_gates)]
        projsz = kwargs.get('projsz')
        if projsz is not None:
            gates.append(('proj', pytorch_linear(self.char_comp.outsz, projsz)))
            self.char_comp.outsz = projsz
        self.gating_seq = nn.Sequential(OrderedDict(gates))

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


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddings(CharConvEmbeddings):

    def __init__(self, _, **kwargs):
        super(PositionalCharConvEmbeddings, self).__init__(_, **kwargs)

        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        # This could get us in trouble, if in doubt, pick something big
        mxlen = kwargs.get('mxlen', 1000)
        max_timescale = kwargs.get('max_timescale', 1.0e4)

        word_dsz = self.get_dsz()

        log_timescale_increment = math.log(max_timescale) / word_dsz
        inv_timescales = torch.exp(torch.arange(0, word_dsz, 2).float() * -log_timescale_increment)

        pe = torch.zeros(mxlen, word_dsz)
        position = torch.arange(0, mxlen).float().unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * inv_timescales)
        pe[:, 1::2] = torch.cos(position * inv_timescales)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def get_dsz(self):
        return self.char_comp.outsz

    def get_vsz(self):
        return self.vsz

    def forward(self, xch):
        """Add a positional encoding to the embedding, followed by dropout

        :param x: The temporal signal in, to which the positional embeddings are applied
        :return: Embedded output
        """
        xch = super(PositionalCharConvEmbeddings, self).forward(xch) * math.sqrt(self.get_dsz())
        xch = xch + self.pe[:, :xch.size(1)]
        return self.dropout(xch)


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddings(LookupTableEmbeddings):

    def __init__(self, _, **kwargs):
        super(PositionalLookupTableEmbeddings, self).__init__(_, **kwargs)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))
        # This could get us in trouble, if in doubt, pick something big
        mxlen = kwargs.get('mxlen', 1000)
        max_timescale = kwargs.get('max_timescale', 1.0e4)

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
        x = super(PositionalLookupTableEmbeddings, self).forward(x) * math.sqrt(self.dsz)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddings(LookupTableEmbeddings):
    def __init__(self, _, **kwargs):
        super(LearnedPositionalLookupTableEmbeddings, self).__init__(_, **kwargs)
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_embeddings = nn.Embedding(self.mxlen, self.dsz, padding_idx=0)
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.1))

    def forward(self, x):
        T = x.size(1)
        x = super(LearnedPositionalLookupTableEmbeddings, self).forward(x)
        pos = self.pos_embeddings(torch.arange(T, dtype=torch.long, device=x.device)).unsqueeze(0)
        return self.dropout(x + pos)

    def extra_repr(self):
        return 'mxlen=%d' % self.mxlen


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(PyTorchEmbeddings):
    def __init__(self, name, **kwargs):
        super(CharLSTMEmbeddings, self).__init__(name, **kwargs)
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
