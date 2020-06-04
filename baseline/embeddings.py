import eight_mile.embeddings
from eight_mile.embeddings import *
from eight_mile.utils import exporter, optional_params, listify, idempotent_append, is_sequence
from baseline.utils import import_user_module, AddonDownloader, EmbeddingDownloader, DEFAULT_DATA_CACHE
import logging

__all__ = []
__all__.extend(eight_mile.embeddings.__all__)
export = exporter(__all__)
logger = logging.getLogger("mead.layers")


MEAD_LAYERS_EMBEDDINGS = {}
MEAD_LAYERS_EMBEDDINGS_LOADERS = {}


@export
@optional_params
def register_embeddings(cls, name=None):
    """Register a function as a plug-in"""
    if name is None:
        name = cls.__name__

    if name in MEAD_LAYERS_EMBEDDINGS:
        raise Exception(
            "Error: attempt to re-define previously registered handler {} (old: {}, new: {}) in registry".format(
                name, MEAD_LAYERS_EMBEDDINGS[name], cls
            )
        )

    MEAD_LAYERS_EMBEDDINGS[name] = cls

    if hasattr(cls, "load"):
        MEAD_LAYERS_EMBEDDINGS_LOADERS[name] = cls.load
    return cls


@export
def create_embeddings(**kwargs):
    embed_type = kwargs.get("embed_type", "default")
    Constructor = MEAD_LAYERS_EMBEDDINGS.get(embed_type)
    return Constructor(**kwargs)


def load_embeddings_overlay(global_embeddings_settings, embeddings_section, vocab, data_download_cache=DEFAULT_DATA_CACHE, name=None):
    """Creates a set of arbitrary sub-graph, DL-framework-specific embeddings by delegating to wired sub-module.

    As part of this process, we take in an index of embeddings by name, a ``dict`` of ``Counter`` objects (keyed by
    feature name), containing the number of times each token has been seen, and a `features` list which is a
    sub-section of the mead config containing the `embeddings` section for each feature.
    This method's job is to either create a sub-graph from a pretrained model, or to create a new random
    initialized sub-graph, taking into account the input vocabulary counters.  The embeddings model has control
    to determine the actual word indices and sub-graph for the embeddings, both of which are returned from this
    method.  If some sort of feature selection is
    performed, such as low count removal that would be required via the delegated methods

    :param global_embeddings_settings: The embeddings index passed to mead driver
    :param vocabs: A set of known ``Counter``s for each vocabulary consisting of a token key and count for each
    :param features: The `features` sub-section of the mead config
    :return: Returns a ``tuple`` comprised of a ``dict`` of (`feature name`, `Embedding`) and an updated vocab
    """

    # Get the label out of the embeddings section in the features block of mead config
    embed_label = embeddings_section.get('label', embeddings_section.get('labels'))
    if name is None:
        name = embed_label
    # Get the type of embedding out of the embeddings section in the features block of mead config
    embed_type = embeddings_section.get('type', 'default')
    is_stacked = is_sequence(embed_label)
    if is_stacked:
        if embed_type != 'default':
            logger.warning("You have requested a stack of pretrained embeddings but didnt request 'default' or representation")
    # Backwards compat, copy from main block if not present locally
    embeddings_section['unif'] = embeddings_section.get('unif', 0.1)

    # Backwards compat, copy from main block if not present locally
    embeddings_section['keep_unused'] = embeddings_section.get('keep_unused', False)

    # Overlay any backend parameters

    # Also, if we are in eager mode, we might have to place the embeddings explicitly on the CPU
    embeddings_section['cpu_placement'] = bool(embeddings_section.get('cpu_placement', False))
    if embed_label is not None:
        # Allow local overrides to uniform initializer

        embed_labels = listify(embed_label)

        embed_files = []
        for embed_label in embed_labels:

            embeddings_global_config_i = global_embeddings_settings[embed_label]
            if 'type' in embeddings_global_config_i:
                embed_type_i = embeddings_global_config_i['type']
                embed_type = embed_type_i
                if embed_type_i != 'default' and is_stacked:
                    raise Exception("Stacking embeddings only works for 'default' pretrained word embeddings")

            embed_file = embeddings_global_config_i.get('file')
            unzip_file = embeddings_global_config_i.get('unzip', True)
            embed_dsz = embeddings_global_config_i['dsz']
            embed_sha1 = embeddings_global_config_i.get('sha1')
            # Should we grab vocab here too?

            embed_model = embeddings_global_config_i.get('model', {})
            if 'dsz' not in embed_model and not is_stacked:
                embed_model['dsz'] = embed_dsz

            embeddings_section = {**embed_model, **embeddings_section}
            try:
                # We arent necessarily going to get an `embed_file`. For instance, using the HuggingFace
                # models in the Hub addon, the `embed_file` should be downloaded using HuggingFace's library,
                # not by us.  In this case we want it to be None and we dont want to download it
                if embed_file:
                    embed_file = EmbeddingDownloader(embed_file, embed_dsz, embed_sha1, data_download_cache, unzip_file=unzip_file).download()
                    embed_files.append(embed_file)
                else:
                    embed_files.append(None)
            except Exception as e:
                if is_stacked:
                    raise e
                logger.warning(f"We were not able to download {embed_file}, passing to the addon")
                embed_files.append(embed_file)
        # If we have stacked embeddings (which only works with `default` model, we need to pass the list
        # If not, grab the first item
        embed_file = embed_files if is_stacked else embed_files[0]
        embedding_bundle = load_embeddings(name, embed_file=embed_file, known_vocab=vocab, embed_type=embed_type,
                                           data_download_cache=data_download_cache,
                                           **embeddings_section)

    else:  # if there is no label given, assume we need random initialization vectors
        dsz = embeddings_section.pop('dsz')
        embedding_bundle = load_embeddings(name,
                                           dsz=dsz,
                                           known_vocab=vocab,
                                           embed_type=embed_type,
                                           data_download_cache=data_download_cache,
                                           **embeddings_section)

    return embedding_bundle

@export
def load_embeddings(name, **kwargs):
    """This method negotiates loading an embeddings sub-graph AND a corresponding vocabulary (lookup from word to int)

    Embeddings and their addons may be downloaded from an http `GET` either via raw URL or using hub notation
    (hub:v1:embeddings/hub:v1:addons)

    This function behaves differently depending on its keyword arguments and the `embed_type`.
    If the registered embeddings class contains a load method on it and we are given an `embed_file`,
    we will assume that we need to load that file, and that the embeddings object wants its own load function used
    for that.  This would be typical, e.g, for a user-defined sub-graph LM.

    For cases where no `embed_file` is provided and there is a `create` method on this class, we  assume that the user
    wants us to build a VSM (`baseline.embeddings.PretrainedEmbeddingsModel`) ourselves, and call
    their create function, which will take in this VSM.

    The VSM is then used to provide the vocabulary back, and the `create` function invokes the class constructor
    with the sub-parts of VSM required to build the graph.

    If there is no create method provided, and there is no load function provided, we simply invoke the
    registered embeddings' constructor with the args, and assume there is a `get_vocab()` method on the
    provided implementation

    :param name: A unique string name for these embeddings
    :param kwargs:

    :Keyword Arguments:
    * *embed_type*  The key identifying the embedding type in the registry
    :return:
    """
    embed_type = kwargs.pop("embed_type", "default")
    # Dynamically load a module if its needed
    for module in listify(kwargs.get('module', kwargs.get('modules', []))):
        import_user_module(module, kwargs.get('data_download_cache'))
    embeddings_cls = MEAD_LAYERS_EMBEDDINGS[embed_type]

    filename = kwargs.get("embed_file")

    # If the embedding model has a load function, defer all the work to that.  Basically just pass the kwargs in
    # and let it do its magic
    if hasattr(embeddings_cls, "load") and filename is not None:
        model = embeddings_cls.load(filename, **kwargs)
        return {"embeddings": model, "vocab": model.get_vocab()}
    # If there isnt a load function, there must be a create() function where the first arg is a type of
    # EmbeddingsModel
    elif hasattr(embeddings_cls, "create"):
        unif = kwargs.pop("unif", 0.1)
        known_vocab = kwargs.pop("known_vocab", None)
        keep_unused = kwargs.pop("keep_unused", False)
        normalize = kwargs.pop("normalized", False)
        preserve_vocab_indices = bool(kwargs.get('preserve_vocab_indices', False))

        # if there is no filename, use random-init model
        if filename is None:
            dsz = kwargs.pop("dsz")
            model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=unif, counts=not preserve_vocab_indices)
        # If there, is use the PretrainedEmbeddingsModel loader
        else:
            if is_sequence(filename):
                model = PretrainedEmbeddingsStack(
                    listify(filename),
                    known_vocab=known_vocab,
                    normalize=normalize,
                    counts=not preserve_vocab_indices,
                    **kwargs
                )
            else:
                model = PretrainedEmbeddingsModel(
                    filename,
                    known_vocab=known_vocab,
                    unif_weight=unif,
                    keep_unused=keep_unused,
                    normalize=normalize,
                    **kwargs,
                )

        # Then call create(model, name, **kwargs)
        return {"embeddings": embeddings_cls.create(model, name, **kwargs), "vocab": model.get_vocab()}
    # If we dont have a load function, but filename is none, we should just instantiate the class
    model = embeddings_cls(name, **kwargs)
    return {"embeddings": model, "vocab": model.get_vocab()}

