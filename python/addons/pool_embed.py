import tensorflow as tf
from eight_mile.utils import Offsets
from eight_mile.embeddings import register_embeddings
from eight_mile.tf.embeddings import LookupTableEmbeddings


@register_embeddings(name='cbow')
class CBoWEmbeddings(LookupTableEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
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
        self.pooler = lambda x, y: tf.reduce_sum(x, axis=1)

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self._weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return CBoWEmbeddings(self._name,
                              vsz=self.vsz,
                              dsz=self.dsz,
                              scope=self.scope,
                              dropin=self.dropin,
                              finetune=self.finetune,
                              weights=self._weights)

    def encode(self, x=None):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        if x is None:
            x = LookupTableEmbeddings.create_placeholder(self._name)
        self.x = x
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        W = shape[2]

        flat = tf.reshape(x, [-1, W])
        emb = super().encode(flat)

        lengths = tf.reduce_sum(tf.cast(tf.equal(flat, Offsets.PAD), tf.int32), axis=1)

        pooled = self.pooler(emb, lengths)

        return tf.reshape(pooled, [B, T, self.dsz])


@register_embeddings(name='max-pool')
class MaxBoWEmbeddings(CBoWEmbeddings):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.pooler = lambda x, y: tf.reduce_max(x, axis=1)


@register_embeddings(name='mean-pool')
class MeanBoWEmbeddings(CBoWEmbeddings):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.pooler = lambda x, y: tf.reduce_sum(x, axis=1) / tf.cast(tf.expand_dims(y, -1), tf.float32)
