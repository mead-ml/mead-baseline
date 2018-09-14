import tensorflow as tf
from baseline.tf.tfy import embed, pool_chars


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

    @classmethod
    def create_placeholder(cls, name):
        pass


class TensorFlowTokenEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None], name=name)

    def __init__(self, model, name, **kwargs):
        super(TensorFlowTokenEmbeddings, self).__init__()
        self.model = model
        self.finetune = kwargs.get('finetune', True)
        self.scope = kwargs.get('scope', 'LUT')
        self.name = name
        self.x = kwargs.get(self.name, self.create_placeholder(name))

    def get_vsz(self):
        return self.model.get_vsz()

    def get_dsz(self):
        return self.model.get_dsz()

    def encode(self):
        return embed(self.x,
                     len(self.model.vocab),
                     self.get_dsz(),
                     tf.constant_initializer(self.model.weights, dtype=tf.float32, verify_shape=True),
                     self.finetune,
                     self.scope)

    def save_md(self, model_file):
        #self.model.save(model_file, ignore_weights=True)
        pass

    @staticmethod
    def load():
        pass


class TensorFlowCharConvEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, model, name, **kwargs):
        super(TensorFlowCharConvEmbeddings)
        self.model = model
        self.params = kwargs
        self.scope = kwargs.get('scope', 'CharLUT')
        self.wsz = None
        self.name = name
        self.xch = kwargs.get(self.name, tf.placeholder(tf.int32, [None, None, None], name=self.name))

    def encode(self):
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable("Wch",
                                  initializer=tf.constant_initializer(self.model.weights, dtype=tf.float32, verify_shape=True),
                                  shape=[len(self.model.vocab), self.model.dsz], trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.model.dsz]))
            char_comp, self.wsz = pool_chars(self.xch, Wch, ech0, self.model.dsz, **self.params)
            return char_comp

    def get_vsz(self):
        return self.model.get_vsz()

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz


def tf_embeddings(in_embeddings_obj, name, DefaultType=TensorFlowTokenEmbeddings, **kwargs):
    if isinstance(in_embeddings_obj, TensorFlowEmbeddings):
        return in_embeddings_obj
    else:
        return DefaultType(in_embeddings_obj, name, **kwargs)
