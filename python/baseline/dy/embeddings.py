import math
from itertools import chain
import numpy as np
from baseline.dy.dynety import ParallelConv, HighwayConnection, SkipConnection, Linear
from baseline.utils import export
from baseline.embeddings import register_embeddings
import dynet as dy
__all__ = []
exporter = export(__all__)


@exporter
class DyNetEmbeddings(object):

    def __init__(self, pc):
        super(DyNetEmbeddings).__init__()
        self.pc = pc

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def encode(self, x):
        pass

    @classmethod
    def create(cls, model, name, **kwargs):
        """Instantiate this sub-graph from the generalized representation from `baseline.w2v`

        :param name: The name of the embeddings
        :param model: The `baseline.w2v` model
        :param kwargs:
        :return:
        """
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


@register_embeddings(name='default')
class LookupTableEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        super(LookupTableEmbeddings, self).__init__(kwargs['pc'])
        self.finetune = kwargs.get('finetune', True)
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.batched = kwargs.get('batched', False)
        embedding_weight = np.reshape(kwargs['weights'], (self.vsz, 1, self.dsz))
        self.lookup = dy.lookup_batch if self.batched else dy.lookup
        self.embeddings = self.pc.lookup_parameters_from_numpy(embedding_weight, name=name)

    def encode(self, x):
        """Encode a sequence.

        :param input_: List[List[int]] (batched) or List[int] (normal)
            When batched the input should be a list over timesteps of lists of
            words (over a batch) (T, B). Otherwise it is a list of words over time (T)

        Returns:
            dy.Expression ((T, H), B) if dense (useful for conv encoders)
            List[dy.Expression] otherwise (used for RNNs)
        """
        embedded = [self.lookup(self.embeddings, v, self.finetune) for v in x]
        return dy.concatenate(embedded, d=0)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz


@register_embeddings(name="positional")
class PositionalLookupTableEmbeddings(DyNetEmbeddings):
    def __init__(self, name, **kwargs):
        super(PositionalLookupTableEmbeddings, self).__init__(kwargs['pc'])
        self.vsz = int(kwargs.get('vsz'))
        self.dsz = int(kwargs.get('dsz'))
        self.dropout = float(kwargs.get('dropout', 0.1))
        mxlen = int(kwargs.get('mxlen', 1000))
        max_timescale = float(kwargs.get('max_timescale', 1e4))
        log_timescale_inc = math.log(max_timescale) / self.dsz
        inv_timescale = np.exp(np.arange(0, self.dsz, 2) * -log_timescale_inc)
        self.embeddings = LookupTableEmbeddings(name, **kwargs)
        pe = np.zeros((mxlen, self.dsz))
        position = np.expand_dims(np.arange(mxlen), 1)
        pe[:, 0::2] = np.sin(position * inv_timescale)
        pe[:, 1::2] = np.cos(position * inv_timescale)
        self.pe = pe

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x, train):
        embedded = self.embeddings.encode(x)
        embedded = embedded * math.sqrt(self.dsz)
        print(embedded.dim())
        ((seq_len, _), _) = embedded.dim()
        embedded = embedded + dy.inputTensor(self.pe[:seq_len])
        embedded = dy.dropout(embedded, self.dropout) if train else embedded
        return embedded


@register_embeddings(name='char-conv')
class CharConvEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__(kwargs['pc'])
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.batched = kwargs.get('batched', False)
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.weights = kwargs.get('weights')
        weights = kwargs.get('weights')
        embedding_weight = np.reshape(weights, (self.vsz, 1, self.dsz))
        self.embeddings = self.pc.lookup_parameters_from_numpy(embedding_weight, name=name)
        filtsz = kwargs.get('cfiltsz', [3])
        gate = kwargs.get('gating', 'skip')
        num_gates = kwargs.get('num_gates', 1)
        max_feat = kwargs.get('max_feat', 200)
        nfeat_factor = kwargs.get('nfeat_factor')
        cmotsz = kwargs.get('wsz', 30)
        self.pool, self.wsz = self._create_char_comp(filtsz, cmotsz, self.dsz, gate, num_gates, max_feat, nfeat_factor)
        self.lookup = dy.lookup_batch #if self.batched else dy.lookup

    def _create_char_comp(self, filtsz, cmotsz, cdsz, gate, num_gates, max_feat, nfeat_factor):
        if nfeat_factor is not None:
            cmotsz = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
            cmotsz_total = sum(cmotsz)
        else:
            cmotsz_total = cmotsz * len(filtsz)
        parallel_conv = ParallelConv(filtsz, cmotsz, cdsz, self.pc)
        gate = HighwayConnection if gate.startswith('highway') else SkipConnection
        funcs = [Linear(cmotsz_total, cmotsz_total, self.pc, name="linear-{}".format(i)) for i in range(num_gates)]
        gating = gate(funcs, cmotsz_total, self.pc)

        def call(input_):
            x = parallel_conv(input_)
            return gating(x)

        return call, cmotsz_total

    def get_dsz(self):
        return self.wsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x):
        xch = x.transpose(0, 2, 1)
        W, T, B = x.shape
        xch = x.reshape(W, -1)
        # W x (T x B)
        embedded = [self.lookup(self.embeddings, v, self.finetune) for v in xch]
        embed_chars_vec = dy.concatenate(embedded)
        embed_chars_vec = dy.reshape(embed_chars_vec, (W, self.dsz), T*B)
        pooled_chars = self.pool(embed_chars_vec)
        pooled_chars = dy.reshape(pooled_chars, (self.wsz, T), B)
        # Back to T x W x B
        pooled_chars = dy.transpose(pooled_chars)
        return pooled_chars
