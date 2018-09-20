import tensorflow as tf
from baseline.tf.tfy import embed, pool_chars
from baseline.utils import write_json


class TensorFlowEmbeddings(object):

    def __init__(self):
        pass

    def get_dsz(self):
        pass

    def get_vsz(self):
        pass

    def encode(self):
        pass

    def save_md(self):
        pass

    def get_vocab(self):
        pass

    @classmethod
    def create_placeholder(cls, name):
        pass

    @classmethod
    def create_from_embeddings(cls, name, model, **kwargs):
        return cls(name, vsz=model.vsz, dsz=model.dsz, vocab=model.vocab, weights=model.weights, **kwargs)


class TensorFlowTokenEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)

    def __init__(self, name, **kwargs):
        super(TensorFlowTokenEmbeddings, self).__init__()
        self.vocab = kwargs.get('vocab')
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/LUT'.format(self.name))
        self.x = kwargs.get(self.name, self.create_placeholder(name))
        self.weights = kwargs.get('weights')

    def get_vocab(self):
        return self.vocab

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.dsz

    def encode(self):
        return embed(self.x,
                     len(self.vocab),
                     self.get_dsz(),
                     tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                     self.finetune,
                     self.scope)

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz(), 'vocab': self.get_vocab()}, target)

    def _get_weights(self, sess):
        weights = sess.run('{}/W:0'.format(self.scope))
        return weights


class TensorFlowCharConvEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(TensorFlowCharConvEmbeddings, self).__init__()
        self.vocab = kwargs.get('vocab')
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
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz(), 'vocab': self.get_vocab()}, target)

    def encode(self):
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable("Wch",
                                  initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                  shape=[len(self.vocab), self.dsz], trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
            char_comp, self.wsz = pool_chars(self.xch, Wch, ech0, self.dsz, **self.params)
            return char_comp

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz

    def get_vocab(self):
        return self.vocab



def tf_embeddings(in_embeddings_obj, name, DefaultType=TensorFlowTokenEmbeddings, **kwargs):
    if isinstance(in_embeddings_obj, TensorFlowEmbeddings):
        return in_embeddings_obj
    else:
        return DefaultType.create_from_embeddings(name, in_embeddings_obj, **kwargs)
