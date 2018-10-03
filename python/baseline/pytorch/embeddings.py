from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
from baseline.utils import load_user_embeddings, create_user_embeddings
import torch.nn as nn
from collections import OrderedDict
from baseline.pytorch.torchy import (pytorch_embedding,
                                     ParallelConv,
                                     pytorch_linear,
                                     pytorch_activation,
                                     SkipConnection,
                                     Highway)


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
    def create(cls, model, **kwargs):
        return cls(model, **kwargs)


# TODO: Make these constructors more like TF, DyNet
class LookupTableEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, model, **kwargs):
        super(LookupTableEmbeddings, self).__init__()
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


class CharConvEmbeddings(nn.Module, PyTorchEmbeddings):

    def __init__(self, model, **kwargs):
        super(CharConvEmbeddings, self).__init__()
        self.embeddings = pytorch_embedding(model)
        self.vsz = model.get_vsz()
        char_filtsz = kwargs.get('cfiltsz', [3])
        char_hsz = kwargs.get('wsz', 30)
        activation_type = kwargs.get('activation', 'tanh')
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = ParallelConv(model.get_dsz(), char_hsz, char_filtsz, activation_type, pdrop)
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


# If the embeddings are listed here, than we need to use PretrainedEmbeddingsModel
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
        return {'embeddings': create_fn(model, **kwargs), 'vocab': model.get_vocab()}
    print('loading user module')
    return load_user_embeddings(filename, name, known_vocab, **kwargs)


def create_embeddings(dsz, name, known_vocab=None, **kwargs):

    embed_type = kwargs.pop('embed_type', 'default')
    create_fn = BASELINE_EMBEDDING_MODELS.get(embed_type)

    if create_fn is not None:
        model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=kwargs.pop('unif', 0))
        return {'embeddings': create_fn(model, **kwargs), 'vocab': model.get_vocab()}

    print('loading user module')
    return create_user_embeddings(dsz, name, known_vocab, **kwargs)
