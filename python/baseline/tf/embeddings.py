import tensorflow as tf
from baseline.tf.tfy import embed, pool_chars
from baseline.utils import write_json, load_user_embeddings, create_user_embeddings
from baseline.w2v import PretrainedEmbeddingsModel, RandomInitVecModel
import numpy as np


class TensorFlowEmbeddings(object):
    """This provides a base for TensorFlow embeddings sub-graphs

    """
    def __init__(self):
        """Constructor
        """
        pass

    def get_dsz(self):
        """Get the number of output dimension of this operation

        :return:
        """
        pass

    def get_vsz(self):
        """Get the number of words (including <PAD>) in the vocabulary

        :return:
        """
        pass

    def encode(self, x=None):
        """This instantiates the sub-graph for this object and returns the output node

        :return:
        """
        pass

    def save_md(self):
        """Save the meta-data associated with this object, namely the `vsz` and `dsz`

        :return:
        """
        pass

    @classmethod
    def create_placeholder(cls, name):
        """Create a placeholder with name `name`

        :param name: (``str``) The name of the placeholder
        :return: The placeholder
        """
        pass

    @classmethod
    def create(cls, model, name, **kwargs):
        """Instantiate this sub-graph from the generalized representation from `baseline.w2v`

        :param name: The name of the embeddings
        :param model: The `baseline.w2v` model
        :param kwargs:
        :return:
        """
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


class LookupTableEmbeddings(TensorFlowEmbeddings):
    """Provide "classic" Lookup-Table based word embeddings

    """

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)

    def __init__(self, name, **kwargs):
        """Create a lookup-table based embedding.

        :param name: The name of the feature/placeholder, and a key for the scope
        :param kwargs:

        :Keyword Arguments: See below
        * *vsz* -- (``int``) this is the vocabulary (input) size of the lookup table
        * *dsz* -- (``int``) the output dimension size of this embedding
        * *finetune* -- (``bool``) (default is `True`) should we allow the sub-graph to learn updated weights
        * *weights* -- (``numpy.ndarray``) Optional `vsz x dsz` weight matrix for initialization
        * *scope* -- (``str``) An optional variable scope, by default it will be `{name}/LUT`
        * *unif* -- (``float``) (defaults to `0.1`) If the weights should be created, what is the random initialization range
        """
        super(LookupTableEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/LUT'.format(self.name))
        self.x = kwargs.get(self.name, self.create_placeholder(name))
        self.weights = kwargs.get('weights')
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def encode(self, x=None):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        if x is None:
            x = self.x
        return embed(x,
                     self.vsz,
                     self.dsz,
                     tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                     self.finetune,
                     self.scope)

    def save_md(self, target):
        """Save the metadata associated with this embedding as a JSON file

        :param target: The name of the output file
        :return:
        """
        write_json({'vsz': self.vsz, 'dsz': self.dsz}, target)


class CharBoWEmbeddings(TensorFlowEmbeddings):
    """Bag of character embeddings, sum char embeds, so in this case `wsz == dsz`

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharBoWEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharBoWLUT'.format(self.name))
        self.weights = kwargs.get('weights')
        self.params = kwargs
        self.wsz = None
        self.xch = kwargs.get(self.name, tf.placeholder(tf.int32, [None, None, None], name=self.name))

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self, x=None):
        if x is None:
            x = self.x
        return tf.reduce_sum(embed(x,
                                   self.get_vsz(),
                                   self.get_dsz(),
                                   tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                   self.finetune,
                                   self.scope), axis=-1, keep_dims=False)


class CharConvEmbeddings(TensorFlowEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharLUT'.format(self.name))
        self.weights = kwargs.get('weights')
        self.params = kwargs
        self.wsz = None
        self.x = kwargs.get(self.name, tf.placeholder(tf.int32, [None, None, None], name=self.name))

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self, x=None):
        if x is None:
            x = self.x
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable("Wch",
                                  initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                  shape=[self.vsz, self.dsz], trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
            char_comp, self.wsz = pool_chars(x, Wch, ech0, self.dsz, **self.params)
            return char_comp

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz


# If the embeddings are listed here, than we need to use PretrainedEmbeddingsModel
# TODO: add/test CBoW to this registry
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
        return {'embeddings': create_fn(model, name, **kwargs), 'vocab': model.get_vocab()}
    print('loading user module')
    return load_user_embeddings(filename, name, known_vocab, **kwargs)


def create_embeddings(dsz, name, known_vocab=None, **kwargs):

    embed_type = kwargs.pop('embed_type', 'default')
    create_fn = BASELINE_EMBEDDING_MODELS.get(embed_type)

    if create_fn is not None:
        model = RandomInitVecModel(dsz, known_vocab=known_vocab, unif_weight=kwargs.pop('unif', 0))
        return {'embeddings': create_fn(model, name, **kwargs), 'vocab': model.get_vocab()}

    print('loading user module')
    return create_user_embeddings(dsz, name, known_vocab, **kwargs)
