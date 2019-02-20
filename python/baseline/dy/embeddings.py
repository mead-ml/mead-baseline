import math
from itertools import chain
import numpy as np
from baseline.dy.dynety import ParallelConv, HighwayConnection, SkipConnection, Linear, DynetLayer, rnn_forward_with_state
from baseline.utils import export, Offsets
from baseline.embeddings import register_embeddings
import dynet as dy
__all__ = []
exporter = export(__all__)


@exporter
class DyNetEmbeddings(DynetLayer):

    def __init__(self, pc):
        super(DyNetEmbeddings, self).__init__(pc)

    def get_vsz(self):
        pass

    def get_dsz(self):
        pass

    def encode(self, x, train=False):
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
        pc = kwargs['pc'].add_subcollection(name=kwargs.get('name', 'lookup'))
        super(LookupTableEmbeddings, self).__init__(pc)
        self.finetune = kwargs.get('finetune', True)
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.batched = kwargs.get('batched', False)
        weight = kwargs.get('weights')
        if weight is None:
            unif = kwargs.get('unif', 0.1)
            weight = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        embedding_weight = np.reshape(weight, (self.vsz, 1, self.dsz))
        self.lookup = dy.lookup_batch if self.batched else dy.lookup
        self.embeddings = self.pc.lookup_parameters_from_numpy(embedding_weight, name=name)

    def encode(self, x, train=False):
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
        pc = kwargs['pc'].add_subcollection(name=kwargs.get('name', 'positional'))
        super(PositionalLookupTableEmbeddings, self).__init__(pc)
        self.vsz = int(kwargs.get('vsz'))
        self.dsz = int(kwargs.get('dsz'))
        self.dropout = float(kwargs.get('dropout', 0.1))
        mxlen = int(kwargs.get('mxlen', 1000))
        max_timescale = float(kwargs.get('max_timescale', 1e4))
        log_timescale_inc = math.log(max_timescale) / self.dsz
        inv_timescale = np.exp(np.arange(0, self.dsz, 2) * -log_timescale_inc)
        kwargs['pc'] = self.pc
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

    def encode(self, x, train=False):
        embedded = self.embeddings.encode(x)
        embedded = embedded * math.sqrt(self.dsz)
        ((seq_len, _), _) = embedded.dim()
        embedded = embedded + dy.inputTensor(self.pe[:seq_len])
        embedded = dy.dropout(embedded, self.dropout) if train else embedded
        return embedded


@register_embeddings(name="learned-positional")
class LearnedPositionalLookupTableEmbeddings(DyNetEmbeddings):
    def __init__(self, name, **kwargs):
        pc = kwargs['pc'].add_subcollection(name=kwargs.get('name', 'positional'))
        super(LearnedPositionalLookupTableEmbeddings, self).__init__(pc)
        self.vsz = int(kwargs.get('vsz'))
        self.dsz = int(kwargs.get('dsz'))
        self.dropout = float(kwargs.get('dropout', 0.1))
        mxlen = int(kwargs.get('mxlen', 512))
        kwargs['pc'] = self.pc
        self.embeddings = LookupTableEmbeddings(name, **kwargs)
        kwargs['vsz'] = mxlen
        kwargs.pop('weights', None)
        self.pos_embeddings = LookupTableEmbeddings('learned-positional', **kwargs)


    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x, train=False):
        embedded = self.embeddings.encode(x)
        ((seq_len, _), _) = embedded.dim()
        seq_run = np.reshape(np.arange(seq_len), (seq_len, 1))
        pos_embedded = self.pos_embeddings.encode(seq_run)
        embedded = embedded + pos_embedded
        embedded = dy.dropout(embedded, self.dropout) if train else embedded
        return embedded


@register_embeddings(name='char-conv')
class CharConvEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        pc = kwargs['pc'].add_subcollection(name=kwargs.get('name', 'conv-char'))
        super(CharConvEmbeddings, self).__init__(pc)
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

        def call(input_, train):
            x = parallel_conv(input_)
            return gating(x, train)

        return call, cmotsz_total

    def get_dsz(self):
        return self.wsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x, train=False):
        xch = x.transpose(0, 2, 1)
        W, T, B = x.shape
        xch = x.reshape(W, -1)
        # W x (T x B)
        embedded = [self.lookup(self.embeddings, v, self.finetune) for v in xch]
        embed_chars_vec = dy.concatenate(embedded)
        embed_chars_vec = dy.reshape(embed_chars_vec, (W, self.dsz), T*B)
        pooled_chars = self.pool(embed_chars_vec, train)
        pooled_chars = dy.reshape(pooled_chars, (self.wsz, T), B)
        # Back to T x W x B
        pooled_chars = dy.transpose(pooled_chars)
        return pooled_chars


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        pc = kwargs['pc'].add_subcollection(name=kwargs.get('name', 'char-lstm'))
        super(CharLSTMEmbeddings, self).__init__(pc)
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        weights = kwargs.get('weights')
        self.embeddings = self.pc.lookup_parameters_from_numpy(weights, name=name)
        self.lstmsz = kwargs.get('lstmsz', 50)
        layers = kwargs.get('layers', 1)
        self.pdrop = kwargs.get('pdrop', 0.5)
        self.lookup = dy.lookup_batch
        self.lstm_fwd = dy.LSTMBuilder(layers, self.dsz, self.lstmsz // 2, model=self.pc)
        self.lstm_bwd = dy.LSTMBuilder(layers, self.dsz, self.lstmsz // 2, model=self.pc)

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x, train=False):
        if train:
            self.lstm_fwd.set_dropout(self.pdrop)
            self.lstm_bwd.set_dropout(self.pdrop)
        else:
            self.lstm_fwd.disable_dropout()
            self.lstm_bwd.disable_dropout()

        W, T, B = x.shape
        xch = x.reshape(W, -1)

        word_lens = np.sum(xch != Offsets.PAD, axis=0)

        embed_chars = [self.lookup(self.embeddings, v, self.finetune) for v in xch]

        _, fwd_state = rnn_forward_with_state(self.lstm_fwd, embed_chars, lengths=word_lens)
        _, bwd_state = rnn_forward_with_state(self.lstm_bwd, embed_chars, lengths=word_lens, backward=True)

        state = dy.concatenate([fwd_state[-1], bwd_state[-1]])
        state = dy.transpose(dy.reshape(state, (self.lstmsz, T), batch_size=B))
        return state
