import numpy as np
from baseline.utils import (
    load_user_classifier_model, create_user_classifier_model,
    load_user_tagger_model, create_user_tagger_model,
    load_user_seq2seq_model, create_user_seq2seq_model,
    load_user_lang_model, create_user_lang_model,
    load_user_embeddings,
    lowercase, revlut,
    export, wrapped_partial, is_sequence
)
from baseline.vectorizers import Token1DVectorizer, Char2DVectorizer, Dict1DVectorizer, Dict2DVectorizer

__all__ = []
exporter = export(__all__)


@exporter
class ClassifierModel(object):
    """Text classifier
    
    Provide an interface to DNN classifiers that use word lookup tables.
    """
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

    def classify(self, batch_dict):
        """Classify a batch of text with whatever features the model can use from the batch_dict.
        The indices correspond to get_vocab().get('word', 0)
        
        :param batch_dict: This normally contains `x`, a `BxT` tensor of indices.  Some classifiers and readers may
        provide other features

        :return: A list of lists of tuples (label, value)
        """
        pass

    def get_labels(self):
        """Return a list of labels, where the offset within the list is the location in a confusion matrix, etc.
        
        :return: A list of the labels for the decision
        """
        pass



@exporter
def create_model(known_creators, input_, output_, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default

    For classification and tagging tasks input_ is embeddings and output_ is labels
    For seq2seq tasks input_ is the source embeddings and output_ is the target embeddings

    :param known_creators: Map of baseline creators, keyed by `model_type`, typically a static factory method
    :param input_: parameter for the input, general word vectors
    :param output_: parameter for the output, generally labels or output vectors
    :param kwargs: Anything required to feed the model its parameters
    :return: A newly created model
    """
    model_type = kwargs.get('model_type', 'default')
    creator_fn = known_creators[model_type] if model_type in known_creators else kwargs['task_fn']
    print('Calling model ', creator_fn)
    return creator_fn(input_, output_, **kwargs)

create_classifier_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_classifier_model,
        name='create_classifier_model'
    )
)
create_tagger_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_tagger_model,
        name='create_tagger_model'
    )
)
create_seq2seq_model = exporter(
    wrapped_partial(
        create_model,
        task_fn=create_user_seq2seq_model,
        name='create_seq2seq_model'
    )
)


@exporter
def create_lang_model(known_creators, embeddings, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type in known_creators:
        creator_fn = known_creators[model_type]
        print('Calling baseline model creator ', creator_fn)
        return creator_fn(embeddings, **kwargs)
    return create_user_lang_model(embeddings, **kwargs)


@exporter
def load_model(known_loaders, outname, **kwargs):
    """If `model_type` is given, use it to load an addon model and construct that OW use default

    :param known_loaders: Map of baseline functions to load the model, typically a static factory method
    :param outname The model name to load
    :param kwargs: Anything required to feed the model its parameters
    :return: A restored model
    """
    model_type = kwargs.get('model_type', 'default')
    loader_fn = known_loaders[model_type] if model_type in known_loaders else kwargs['task_fn']
    return loader_fn(outname, **kwargs)

load_classifier_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_classifier_model,
        name='load_classifier_model'
    )
)
load_tagger_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_tagger_model,
        name='load_tagger_model'
    )
)
load_seq2seq_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_seq2seq_model,
        name='load_seq2seq_model'
    )
)
load_lang_model = exporter(
    wrapped_partial(
        load_model,
        task_fn=load_user_lang_model,
        name='load_seq2seq_model'
    )
)


@exporter
class TaggerModel(object):
    """Structured prediction classifier, AKA a tagger
    
    This class takes a temporal signal, represented as words over time, and characters of words
    and generates an output label for each time.  This type of model is used for POS tagging or any
    type of chunking (e.g. NER, POS chunks, slot-filling)
    """
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

    def __init__(self):
        super(LanguageModel, self).__init__()

    def step(self, batch_time, context):
        pass


@exporter
class EncoderDecoderModel(object):

    def save(self, model_base):
        pass

    def __init__(self):
        super(EncoderDecoderModel, self).__init__()

    @staticmethod
    def create(src_vocab, dst_vocab, **kwargs):
        pass

    def create_loss(self):
        pass

    def get_src_vocab(self):
        pass

    def get_dst_vocab(self):
        pass

    @staticmethod
    def load(basename, **kwargs):
        pass

    def run(self, source_dict, **kwargs):
        pass

