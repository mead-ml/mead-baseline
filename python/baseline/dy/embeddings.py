from itertools import chain
import numpy as np
from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
from baseline.dy.dynety import ParallelConv
from baseline.utils import create_user_embeddings, load_user_embeddings
import dynet as dy


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


class LookupTableEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        super(LookupTableEmbeddings, self).__init__(kwargs['pc'])
        self.finetune = kwargs.get('finetune', True)
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.batched = kwargs.get('batched', False)
        self.finetune = kwargs.get('finetune', False)
        #embedding_weight = kwargs['weights']
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


class CharConvEmbeddings(DyNetEmbeddings):

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__(kwargs['pc'])
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.weights = kwargs.get('weights')
        weights = kwargs.get('weights')
        embedding_weight = np.reshape(weights, (self.vsz, 1, self.dsz))
        vsz, dsz = embedding_weight.shape
        embedding_weight = np.reshape(embedding_weight, (vsz, 1, dsz))
        self.embeddings = self.pc.lookup_parameters_from_numpy(embedding_weight, name=name)
        filtsz = kwargs.get('cfiltsz', [3])
        cmotsz = kwargs.get('wsz', 30)
        self.wsz = len(filtsz) * cmotsz
        self.pool = ParallelConv(filtsz, cmotsz, self.dsz, self.pc, name=self.name)
        self.lookup = dy.lookup_batch if self.batched else dy.lookup

    def get_dsz(self):
        return self.wsz

    def get_vsz(self):
        return self.vsz

    def encode(self, x):
        xch = x.transpose(2, 0, 1)
        W, T, B = xch.shape
        xch = xch.reshape(W, -1)
        # W x (T x B)
        embedded = [self.lookup(self.embeddings, v, self.finetune) for v in xch]
        embed_chars_vec = dy.concatenate(embedded)
        embed_chars_vec = dy.reshape(embed_chars_vec, (W, self.char_dsz), T*B)
        pooled_chars = self.pool_chars(embed_chars_vec, 1)
        pooled_chars = dy.reshape(pooled_chars, (self.char_word_sz, T), B)
        # Back to T x W x B
        pooled_chars = dy.transpose(pooled_chars)
        return self.pool(pooled_chars)


BASELINE_EMBEDDING_MODELS = {
    'default': LookupTableEmbeddings.create,
    'char-conv': CharConvEmbeddings.create
}


def load_embeddings(filename, name, known_vocab=None, **kwargs):

    embed_type = kwargs.pop('embed_type', 'default')
    create_fn = BASELINE_EMBEDDING_MODELS.get(embed_type)

    if create_fn is not None:
        model = PretrainedEmbeddingsModel(filename,
                                          known_vocab=known_vocab,
                                          unif_weight=kwargs.pop('unif', 0),
                                          keep_unused=kwargs.pop('keep_unused', False),
                                          normalize=kwargs.pop('normalized', False), **kwargs)
        return {'embeddings': create_fn(model, name, **kwargs), 'vocab': model.get_vocab()}
    print('loading user module')
    return load_user_embeddings(filename, name, known_vocab, **kwargs)


def create_embeddings(dsz, name, known_vocab=None, **kwargs):

    embed_type = kwargs.pop('embed_type', 'default')
    create_fn = BASELINE_EMBEDDING_MODELS.get(embed_type)

    if create_fn is not None:
        model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=kwargs.pop('unif', 0))
        return {'embeddings': create_fn(model, name, **kwargs), 'vocab': model.get_vocab()}

    print('loading user module')
    return create_user_embeddings(dsz, name, known_vocab, **kwargs)