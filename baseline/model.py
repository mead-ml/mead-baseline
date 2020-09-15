import logging
import numpy as np
from baseline.utils import (
    exporter, optional_params, listify, register, import_user_module, read_json
)

__all__ = []
export = exporter(__all__)
logger = logging.getLogger('baseline')

BASELINE_MODELS = {}
BASELINE_LOADERS = {}


@export
@optional_params
def register_model(cls, task, name=None):
    """Register a function as a plug-in"""
    if name is None:
        name = cls.__name__

    names = listify(name)

    if task not in BASELINE_MODELS:
        BASELINE_MODELS[task] = {}

    if task not in BASELINE_LOADERS:
        BASELINE_LOADERS[task] = {}

    if hasattr(cls, 'create'):
        def create(*args, **kwargs):
            return cls.create(*args, **kwargs)
    else:
        def create(*args, **kwargs):
            return cls(*args, **kwargs)

    for alias in names:
        if alias in BASELINE_MODELS[task]:
            raise Exception('Error: attempt to re-define previously registered handler {} (old: {}, new: {}) for task {} in registry'.format(alias, BASELINE_MODELS[task], cls, task))

        BASELINE_MODELS[task][alias] = create

        if hasattr(cls, 'load'):
            BASELINE_LOADERS[task][alias] = cls.load
    return cls


@export
def create_model_for(activity, **kwargs):
    model_type = kwargs.get('type', kwargs.get('model_type', 'default'))
    creator_fn = BASELINE_MODELS[activity][model_type]
    logger.info('Calling model %s', creator_fn)
    input_ = kwargs.pop('features', None)
    output_ = kwargs.pop('labels', None)
    if output_ is not None:
        return creator_fn(input_, output_, **kwargs)
    return creator_fn(input_, **kwargs)


@export
def create_model(embeddings, labels, **kwargs):
    return create_model_for('classify', features=embeddings, labels=labels, **kwargs)


@export
def create_tagger_model(embeddings, labels, **kwargs):
    return create_model_for('tagger', features=embeddings, labels=labels, **kwargs)



BASELINE_SEQ2SEQ_ENCODERS = {}

@export
@optional_params
def register_encoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ENCODERS, name, 'encoder')


BASELINE_SEQ2SEQ_DECODERS = {}

@export
@optional_params
def register_decoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_DECODERS, name, 'decoder')


BASELINE_SEQ2SEQ_ARC_POLICY = {}

@export
@optional_params
def register_arc_policy(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ARC_POLICY, name, 'decoder')


@export
def create_seq2seq_decoder(tgt_embeddings, **kwargs):
    decoder_type = kwargs.get('decoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_DECODERS.get(decoder_type)
    return Constructor(tgt_embeddings, **kwargs)


@export
def create_seq2seq_encoder(**kwargs):
    encoder_type = kwargs.get('encoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ENCODERS.get(encoder_type)
    return Constructor(**kwargs)


@export
def create_seq2seq_arc_policy(**kwargs):
    arc_type = kwargs.get('arc_policy_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ARC_POLICY.get(arc_type)
    return Constructor()


@export
def create_seq2seq_model(embeddings, labels, **kwargs):
    return create_model_for('seq2seq', embeddings, labels, **kwargs)


@export
def create_lang_model(embeddings, **kwargs):
    return create_model_for('lm', embeddings, None, **kwargs)


@export
def load_model_for(activity, filename, **kwargs):
    # Sniff state to see if we need to import things
    state = read_json('{}.state'.format(filename))
    if 'hub_modules' in state:
        for hub_module in state['hub_modules']:
            import_user_module(hub_module)
    # There won't be a module for pytorch (there is no state file to load).
    if 'module' in state:
        import_user_module(state['module'])
    # Allow user to override model type (for back compat with old api), backoff
    # to the model type in the state file or to default.
    # TODO: Currently in pytorch all models are always reloaded with the load
    # classmethod with a default model class. This is fine given how simple pyt
    # loading is but it could cause problems if a model has a custom load
    model_type = kwargs.get('type', kwargs.get('model_type', state.get('type', state.get('model_type', 'default'))))
    creator_fn = BASELINE_LOADERS[activity][model_type]
    logger.info('Calling model %s', creator_fn)
    return creator_fn(filename, **kwargs)


@export
def load_model(filename, **kwargs):
    return load_model_for('classify', filename, **kwargs)


@export
def load_tagger_model(filename, **kwargs):
    return load_model_for('tagger', filename, **kwargs)


@export
def load_seq2seq_model(filename, **kwargs):
    return load_model_for('seq2seq', filename, **kwargs)


@export
def load_lang_model(filename, **kwargs):
    return load_model_for('lm', filename, **kwargs)


@export
class ClassifierModel:
    """Text classifier

    Provide an interface to DNN classifiers that use word lookup tables.
    """
    task_name = 'classify'

    def __init__(self, *args, **kwargs):
        super().__init__()

    def save(self, basename):
        """Save this model out

        :param basename: Name of the model, not including suffixes
        :return: None
        """
        pass

    @classmethod
    def load(cls, basename, **kwargs):
        """Load the model from a basename, including directory

        :param basename: Name of the model, not including suffixes
        :param kwargs: Anything that is useful to optimize experience for a specific framework
        :return: A newly created model
        """
        pass

    def predict(self, batch_dict):
        """Classify a batch of text with whatever features the model can use from the batch_dict.
        The indices correspond to get_vocab().get('word', 0)

        :param batch_dict: This normally contains `x`, a `BxT` tensor of indices.  Some classifiers and readers may
        provide other features

        :return: A list of lists of tuples (label, value)
        """
        pass

    # deprecated: use predict
    def classify(self, batch_dict):
        logger.warning('`classify` is deprecated, use `predict` instead.')
        return self.predict(batch_dict)

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.

        :return: A list of the labels for the decision
        """
        pass


@export
class DependencyParserModel:
    """Text classifier

    Provide an interface to DNN dependency parsers
    """
    task_name = 'deps'

    def __init__(self, *args, **kwargs):
        super().__init__()

    def save(self, basename):
        """Save this model out

        :param basename: Name of the model, not including suffixes
        :return: None
        """

    @classmethod
    def load(cls, basename, **kwargs):
        """Load the model from a basename, including directory

        :param basename: Name of the model, not including suffixes
        :param kwargs: Anything that is useful to optimize experience for a specific framework
        :return: A newly created model
        """

    def predict(self, batch_dict):
        """Parse a batch of text with whatever features the model can use from the batch_dict.

        :param batch_dict: Features for prediction

        :return: A tuple of the arcs (heads) and rels
        """

    def decode(self, batch_dict, **kwargs):
        """Decode a parse.  This finds the best arcs and rels using some criteria or method

        :param batch_dict:
        :param kwargs:
        :return:
        """

    def get_labels(self):
        """Return a dictionary of mappings from class to label.  The primary keys are 'heads' and 'labels'

        :return: A list of the labels for the decision
        """
        pass


@export
class TaggerModel:
    """Structured prediction classifier, AKA a tagger

    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
    task_name = 'tagger'

    def __init__(self):
        super().__init__()

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

    def get_labels(self):
        pass


@export
class LanguageModel(object):

    task_name = 'lm'

    def __init__(self):
        super().__init__()

    @staticmethod
    def load(basename, **kwargs):
        pass

    @classmethod
    def create(cls, embeddings, **kwargs):
        pass

    def predict(self, batch_dict, **kwargs):
        pass


@export
class EncoderDecoderModel(object):

    task_name = 'seq2seq'

    def save(self, model_base):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def load(basename, **kwargs):
        pass

    @classmethod
    def create(cls, src_embeddings, dst_embedding, **kwargs):
        pass

    def create_loss(self):
        pass

    def predict(self, source_dict, **kwargs):
        pass

    # deprecated: use predict
    def run(self, source_dict, **kwargs):
        logger.warning('`run` is deprecated, use `predict` instead.')
        return self.predict(source_dict, **kwargs)
