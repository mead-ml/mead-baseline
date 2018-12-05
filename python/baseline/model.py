import numpy as np
from baseline.utils import (
    export, optional_params, listify, register
)

__all__ = []
exporter = export(__all__)


BASELINE_MODELS = {}
BASELINE_LOADERS = {}


@exporter
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


@exporter
def create_model_for(activity, input_, output_, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    creator_fn = BASELINE_MODELS[activity][model_type]
    print('Calling model ', creator_fn)
    if output_ is not None:
        return creator_fn(input_, output_, **kwargs)
    return creator_fn(input_, **kwargs)


@exporter
def create_model(embeddings, labels, **kwargs):
    return create_model_for('classify', embeddings, labels, **kwargs)


@exporter
def create_tagger_model(embeddings, labels, **kwargs):
    return create_model_for('tagger', embeddings, labels, **kwargs)



BASELINE_SEQ2SEQ_ENCODERS = {}

@exporter
@optional_params
def register_encoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ENCODERS, name, 'encoder')


BASELINE_SEQ2SEQ_DECODERS = {}

@exporter
@optional_params
def register_decoder(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_DECODERS, name, 'decoder')


BASELINE_SEQ2SEQ_ARC_POLICY = {}

@exporter
@optional_params
def register_arc_policy(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_SEQ2SEQ_ARC_POLICY, name, 'decoder')


@exporter
def create_seq2seq_decoder(tgt_embeddings, **kwargs):
    type = kwargs.get('decoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_DECODERS.get(type)
    return Constructor(tgt_embeddings, **kwargs)


@exporter
def create_seq2seq_encoder(**kwargs):
    type = kwargs.get('encoder_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ENCODERS.get(type)
    return Constructor(**kwargs)


@exporter
def create_seq2seq_arc_policy(**kwargs):
    type = kwargs.get('arc_policy_type', 'default')
    Constructor = BASELINE_SEQ2SEQ_ARC_POLICY.get(type)
    return Constructor()


@exporter
def create_seq2seq_model(embeddings, labels, **kwargs):
    return create_model_for('seq2seq', embeddings, labels, **kwargs)


@exporter
def create_lang_model(embeddings, **kwargs):
    return create_model_for('lm', embeddings, None, **kwargs)


@exporter
def load_model_for(activity, filename, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    creator_fn = BASELINE_LOADERS[activity][model_type]
    #print('Calling model ', creator_fn)
    return creator_fn(filename, **kwargs)


@exporter
def load_embeddings(filename, **kwargs):
    return load_model_for('embeddings', filename, **kwargs)

@exporter
def load_model(filename, **kwargs):
    return load_model_for('classify', filename, **kwargs)


@exporter
def load_tagger_model(filename, **kwargs):
    return load_model_for('tagger', filename, **kwargs)


@exporter
def load_seq2seq_model(filename, **kwargs):
    return load_model_for('seq2seq', filename, **kwargs)


@exporter
def load_lang_model(filename, **kwargs):
    return load_model_for('lm', filename, **kwargs)


@exporter
class ClassifierModel(object):
    """Text classifier
    
    Provide an interface to DNN classifiers that use word lookup tables.
    """
    task_name = 'classify'

    def __init__(self):
        super(ClassifierModel, self).__init__()

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
        return self.predict(batch_dict)

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.
        
        :return: A list of the labels for the decision
        """
        pass


@exporter
class TaggerModel(object):
    """Structured prediction classifier, AKA a tagger
    
    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
    task_name = 'tagger'

    def __init__(self):
        super(TaggerModel, self).__init__()

    def save(self, basename):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def predict(self, batch_dict):
        pass

    def get_labels(self):
        pass


@exporter
class LanguageModel(object):

    task_name = 'lm'

    def __init__(self):
        super(LanguageModel, self).__init__()

    @staticmethod
    def load(basename, **kwargs):
        pass

    @classmethod
    def create(cls, embeddings, **kwargs):
        pass

    def predict(self, batch_dict, **kwargs):
        pass


@exporter
class EncoderDecoderModel(object):

    task_name = 'seq2seq'

    def save(self, model_base):
        pass

    def __init__(self, *args, **kwargs):
        super(EncoderDecoderModel, self).__init__()

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
        return self.predict(source_dict, **kwargs)