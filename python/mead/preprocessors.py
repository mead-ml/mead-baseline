from baseline.utils import register, export, optional_params


__all__ = []
exporter = export(__all__)

BASELINE_PREPROCESSORS = {}


@exporter
class Preprocessor(object):
    """
    Generic class for creating vectorizers using tensorflow/pyt ops, to be used for exporting models when the service gets
    a string instead of a vectorized input.
    """

    def __init__(self, feature, vectorizer, index, vocab, **kwargs):
        self.feature = feature
        self.vectorizer = vectorizer
        self.index = index
        self.vocab = vocab

    def preproc(self, tf_example):
        """
        Create a preprocessor chain inside of the tensorflow graph.
        """
        pass


@exporter
@optional_params
def register_preprocessor(cls, name=None):
    """Register a function as a plug-in"""
    return register(cls, BASELINE_PREPROCESSORS, name, 'preprocessor')


@exporter
def create_preprocessors(**kwargs):
    preprocessor_type = kwargs['preprocessor_type']  # fail early
    Constructor = BASELINE_PREPROCESSORS.get(preprocessor_type)
    if Constructor is None:
        raise NotImplementedError('no preproc exporter found for type {}'.format(preprocessor_type))
    return Constructor(**kwargs)
