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
        raise Exception('Error: attempt to re-define previously registered handler {} (old: {}, new: {}) in registry'.format(name, BASELINE_EMBEDDINGS[name], cls))

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
    """This method negotiates loading an embeddings sub-graph AND a corresponding vocabulary (lookup from word to int)

    This function behaves differently depending on its keyword arguments and the `embed_type`.
    If the registered embeddings class contains a load method on it and we are given an `embed_file`,
    we will assume that we need to load that file, and that the embeddings object wants its own load function used
    for that.  This would be typical, e.g, for a user-defined sub-graph LM.

    For cases where no `embed_file` is provided and there is a `create` method on this class, we  assume that the user
    wants us to build a VSM (`baseline.w2v`) ourselves, and call their create function, which will take in this VSM.
    The VSM is then used to provide the vocabulary back, and the `create` function invokes the class constructor
    with the sub-parts of VSM required to build the graph.

    If there is no create method provided, and there is no load function provided, we simply invoke the regsitered embeddings'
    constructor with the args, and assume there is a `get_vocab()` method on the provided implementation

    :param name: (``str``) A unique string name for these embeddings
    :param kwargs:
    :return:
    """
    embed_type = kwargs.pop('embed_type', 'default')
    embeddings_cls = BASELINE_EMBEDDINGS[embed_type]

    filename = kwargs.get('embed_file')

    # If the class has a load function, we are going to use that
    if hasattr(embeddings_cls, 'load') and filename is not None:
        model = embeddings_cls.load(filename, **kwargs)
        return {'embeddings': model, 'vocab': model.get_vocab()}

    elif hasattr(embeddings_cls, 'create'):
        unif = kwargs.pop('unif', 0.1)
        known_vocab = kwargs.pop('known_vocab', None)
        keep_unused = kwargs.pop('keep_unused', False)
        normalize = kwargs.pop('normalized', False)
        # if there is no filename, use random-init model
        if filename is None:
            dsz = kwargs.pop('dsz')
            model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=unif)
        # If there, is use hte pretrain loader
        else:
            model = PretrainedEmbeddingsModel(filename,
                                              known_vocab=known_vocab,
                                              unif_weight=unif,
                                              keep_unused=keep_unused,
                                              normalize=normalize,
                                              **kwargs)

        # Then call create(model, name, **kwargs)
        return {'embeddings': embeddings_cls.create(model, name, **kwargs), 'vocab': model.get_vocab()}
    # If we dont have a load function, but filename is none, we should just instantiate the class
    model = embeddings_cls(name, **kwargs)
    return {'embeddings': model, 'vocab': model.get_vocab()}
