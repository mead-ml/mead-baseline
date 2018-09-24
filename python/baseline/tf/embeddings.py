import tensorflow as tf
from baseline.tf.tfy import embed, pool_chars
from baseline.utils import write_json
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

    def encode(self):
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
    def create_from_embeddings(cls, name, model, **kwargs):
        """Instantiate this sub-graph from the generalized representation from `baseline.w2v`

        :param name: The name of the embeddings
        :param model: The `baseline.w2v` model
        :param kwargs:
        :return:
        """
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


class TensorFlowTokenEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)

    def __init__(self, name, **kwargs):
        super(TensorFlowTokenEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/LUT'.format(self.name))
        self.x = kwargs.get(self.name, self.create_placeholder(name))
        self.weights = kwargs.get('weights')
        if self.weights is None:
            self.weights = np.zeros((self.vsz, self.dsz))

    def encode(self):
        return embed(self.x,
                     self.vsz,
                     self.dsz,
                     tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                     self.finetune,
                     self.scope)

    def save_md(self, target):
        write_json({'vsz': self.vsz, 'dsz': self.dsz}, target)


class TensorFlowCharBoWEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(TensorFlowCharBoWEmbeddings, self).__init__()
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

    def encode(self):
        return tf.reduce_sum(embed(self.x,
                                   self.get_vsz(),
                                   self.get_dsz(),
                                   tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                   self.finetune,
                                   self.scope), axis=-1, keep_dims=False)


class TensorFlowCharConvEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(TensorFlowCharConvEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharLUT'.format(self.name))
        self.weights = kwargs.get('weights')
        self.params = kwargs
        self.wsz = None
        self.xch = kwargs.get(self.name, tf.placeholder(tf.int32, [None, None, None], name=self.name))

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self):
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable("Wch",
                                  initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                  shape=[self.vsz, self.dsz], trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
            char_comp, self.wsz = pool_chars(self.xch, Wch, ech0, self.dsz, **self.params)
            return char_comp

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz


def tf_embeddings(in_embeddings_obj, name, DefaultType=TensorFlowTokenEmbeddings, **kwargs):
    if isinstance(in_embeddings_obj, TensorFlowEmbeddings):
        return in_embeddings_obj
    else:
        return DefaultType.create_from_embeddings(name, in_embeddings_obj, **kwargs)
