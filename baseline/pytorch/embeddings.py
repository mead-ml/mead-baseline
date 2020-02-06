from baseline.embeddings import register_embeddings
from eight_mile.pytorch.embeddings import *


class PyTorchEmbeddingsModel(PyTorchEmbeddings):
    """A subclass of embeddings layers to prep them for registration and creation via baseline.

    In tensorflow this layer handles the creation of placeholders and things like that so the
    embeddings layer can just be tensor in tensor out but in pytorch all it does is strip the
    unused `name` input and register them.
    """
    def __init__(self, _=None, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, model, name, **kwargs):
        kwargs.pop("dsz")
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


@register_embeddings(name='default')
class LookupTableEmbeddingsModel(PyTorchEmbeddingsModel, LookupTableEmbeddings):
    pass


@register_embeddings(name='char-conv')
class CharConvEmbeddingsModel(PyTorchEmbeddingsModel, CharConvEmbeddings):
    pass


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, CharLSTMEmbeddings):
    pass


@register_embeddings(name='char-transformer')
class CharTransformerEmbeddingsModel(PyTorchEmbeddingsModel, CharTransformerEmbeddings):
    pass

@register_embeddings(name='positional')
class PositionalLookupTableEmbeddingsModel(PyTorchEmbeddingsModel, PositionalLookupTableEmbeddings):
    pass


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalLookupTableEmbeddings):
    pass


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddingsModel(PyTorchEmbeddingsModel, PositionalCharConvEmbeddings):
    pass


@register_embeddings(name='learned-positional-char-conv')
class LearnedPositionalCharConvEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalCharConvEmbeddings):
    pass


@register_embeddings(name='positional-char-lstm')
class PositionalCharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, PositionalCharLSTMEmbeddings):
    pass


@register_embeddings(name='learned-positional-char-lstm')
class LearnedPositionalCharLSTMEmbeddingsModel(PyTorchEmbeddingsModel, LearnedPositionalCharLSTMEmbeddings):
    pass
