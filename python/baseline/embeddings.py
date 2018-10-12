import numpy as np
from baseline.utils import (
    export, optional_params
)
from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel

__all__ = []
exporter = export(__all__)


BASELINE_EMBEDDINGS = {}
BASELINE_EMBEDDINGS_LOADERS = {}

@exporter
@optional_params
def register_embeddings(cls, name=None):
    """Register a function as a plug-in"""
    if name is None:
        name = cls.__name__

    if name in BASELINE_EMBEDDINGS:
        raise Exception('Error: attempt to re-defined previously registered handler {} for embeddings in registry'.format(name))

    BASELINE_EMBEDDINGS[name] = cls

    if hasattr(cls, 'load'):
        BASELINE_EMBEDDINGS_LOADERS[name] = cls.load
    return cls


@exporter
def create_embeddings(**kwargs):
    embed_type = kwargs.get('embed_type', 'default')
    Constructor = BASELINE_EMBEDDINGS.get(embed_type)
    return Constructor(**kwargs)


@exporter
def load_embeddings(name, **kwargs):

    embed_type = kwargs.pop('embed_type', 'default')
    known_vocab = kwargs.pop('known_vocab')
    embeddings_cls = BASELINE_EMBEDDINGS[embed_type]

    filename = kwargs.get('embed_file')

    # If the class has a load function, we are going to use that
    if hasattr(embeddings_cls, 'load') and filename is not None:
        model = embeddings_cls.load(filename, **kwargs)
        return {'embeddings': model, 'vocab': model.get_vocab()}
    else:
        # if there is no filename, use random-init model
        if filename is None:
            dsz = kwargs['dsz']
            model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=kwargs.pop('unif', 0))
        # If there, is use hte pretrain loader
        else:
            model = PretrainedEmbeddingsModel(filename,
                                              known_vocab=known_vocab,
                                              unif_weight=kwargs.pop('unif', 0),
                                              keep_unused=kwargs.pop('keep_unused', False),
                                              normalize=kwargs.pop('normalized', False), **kwargs)

        # Then call create(model, name, **kwargs)
        return {'embeddings': embeddings_cls.create(model, name, **kwargs), 'vocab': model.get_vocab()}
    # If we dont have a load function, but filename is none, we should just instantiate the class
    model = embeddings_cls(name, **kwargs)
    return {'embeddings': model, 'vocab': model.get_vocab()}
