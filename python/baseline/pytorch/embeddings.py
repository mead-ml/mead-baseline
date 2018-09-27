from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
from baseline.utils import load_user_embeddings, create_user_embeddings
import torch.nn as nn

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

    @classmethod
    def create(cls, model, **kwargs):
        return cls(model, **kwargs)


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


#def pytorch_embeddings(in_embeddings_obj, DefaultType=PyTorchWordEmbeddings, **kwargs):
#    if isinstance(in_embeddings_obj, PyTorchEmbeddings):
#        return in_embeddings_obj
#    else:
#        return DefaultType(in_embeddings_obj, **kwargs)

"""
def _init_char_encoder(self, char_dsz, char_vec, **kwargs):
        self.cembed = pytorch_embedding(char_vec)
        filtsz = kwargs['cfiltsz']
        cmotsz = kwargs['hsz']

        wchsz = cmotsz * len(filtsz)
        self.highway = nn.Sequential()
        append2seq(self.highway, (
            Highway(wchsz),
            Highway(wchsz)
        ))

        # Width of concat of parallel convs
        return wchsz

    def _char_encoder(self, batch_first_words):
        emb = self.dropout(self.cembed(batch_first_words))
        embeddings = emb.transpose(1, 2).contiguous()
        mots = []
        for conv in self.convs:
            # In Conv1d, data BxCxT, max over time
            conv_out = conv(embeddings)
            mot, _ = conv_out.max(2)
            mots.append(mot)

        mots = torch.cat(mots, 1)
        output = self.highway(mots)
        return self.dropout(output)
"""
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
