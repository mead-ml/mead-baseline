import math
import numpy as np
import tensorflow as tf
from baseline.utils import write_json, Offsets
from baseline.embeddings import register_embeddings
from baseline.tf.tfy import embed, pool_chars, get_shape_as_list, stacked_lstm


class TensorFlowEmbeddings(object):
    """This provides a base for TensorFlow embeddings sub-graphs

    """
    def __init__(self):
        """Constructor
        """
        pass

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        raise NotImplementedError

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

    def __call__(self, x):
        return self.encode(x)

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


@register_embeddings(name='default')
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
        self.weights = kwargs.get('weights')

        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return LookupTableEmbeddings(self.name,
                                     vsz=self.vsz,
                                     dsz=self.dsz,
                                     scope=self.scope,
                                     finetune=self.finetune,
                                     weights=self.weights)

    def encode(self, x=None):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        if x is None:
            x = LookupTableEmbeddings.create_placeholder(self.name)
        self.x = x

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


@register_embeddings(name='char-conv')
class CharConvEmbeddings(TensorFlowEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__()

        self.name = name
        self.scope = kwargs.get('scope', '{}/CharLUT'.format(self.name))
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.weights = kwargs.get('weights')
        self.finetune = kwargs.get('finetune', True)
        self.nfeat_factor = kwargs.get('nfeat_factor', None)
        self.cfiltsz = kwargs.get('cfiltsz', [3])
        self.max_feat = kwargs.get('max_feat', 30)
        self.gating = kwargs.get('gating', 'skip')
        self.num_gates = kwargs.get('num_gates', 1)
        self.activation = kwargs.get('activation', 'tanh')
        self.wsz = kwargs.get('wsz', 30)
        self.weights = kwargs.get('weights', None)

        self.x = None
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return CharConvEmbeddings(name=self.name, vsz=self.vsz, dsz=self.dsz, scope=self.scope,
                                  finetune=self.finetune, nfeat_factor=self.nfeat_factor,
                                  cfiltsz=self.cfiltsz, max_feat=self.max_feat, gating=self.gating,
                                  num_gates=self.num_gates, activation=self.activation, wsz=self.wsz,
                                  weights=self.weights)

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self, x=None):
        if x is None:
            x = CharConvEmbeddings.create_placeholder(self.name)
        self.x = x
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable("Wch",
                                  initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                  shape=[self.vsz, self.dsz], trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
            char_comp, self.wsz = pool_chars(x, Wch, ech0, self.dsz, self.nfeat_factor,
                                  self.cfiltsz, self.max_feat, self.gating, 
                                  self.num_gates, self.activation, self.wsz)
            return char_comp

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        tf.maximum(tf.to_float(num_timescales) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


@register_embeddings(name='positional')
class PositionalLookupTableEmbeddings(LookupTableEmbeddings):

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
        super(PositionalLookupTableEmbeddings, self).__init__(name, **kwargs)
        self.max_timescale = kwargs.get('max_timescale', 1.0e4)

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return PositionalLookupTableEmbeddings(self.name,
                                               vsz=self.vsz,
                                               dsz=self.dsz,
                                               finetune=self.finetune,
                                               scope=self.scope,
                                               weights=self.weights,
                                               max_timescale=self.max_timescale)

    def encode(self, x=None):
        x = super(PositionalLookupTableEmbeddings, self).encode(x) * math.sqrt(self.dsz)
        B, T, C = get_shape_as_list(x)
        signal = get_timing_signal_1d(T, C, min_timescale=1.0, max_timescale=self.max_timescale, start_index=0)
        return x + signal

    @classmethod
    def create(cls, model, name, **kwargs):
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)


@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharLSTMEmbeddings, self).__init__()
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharLUT'.format(self.name))
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.weights = kwargs.get('weights')
        self.lstmsz = kwargs.get('lstmsz', 50)
        self.layers = kwargs.get('layers', 1)
        self.pdrop = kwargs.get('pdrop', 0.5)
        self.rnn_type = kwargs.get('rnn_type', 'blstm')
        self.x = None
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def detached_ref(self):
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method.')
        return CharLSTMEmbeddings(
            name=self.name, vsz=self.vsz, dsz=self.dsz, scope=self.scope,
            finetune=self.finetune, lstmsz=self.lstmsz, layers=self.layers,
            dprop=self.pdrop, rnn_type=self.rnn_type, weights=self.weights,
        )

    def encode(self, x=None):
        if x is None:
            x = CharLSTMEmbeddings.create_placeholder(self.name)
        self.x = x
        with tf.variable_scope(self.scope):
            Wch = tf.get_variable(
                "Wch",
                initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                shape=[self.vsz, self.dsz],
                trainable=True
            )
            ech0 = tf.scatter_update(Wch, tf.constant(Offsets.PAD, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))

            shape = tf.shape(x)
            B = shape[0]
            T = shape[1]
            W = shape[2]
            flat_chars = tf.reshape(x, [-1, W])
            word_lengths = tf.reduce_sum(tf.cast(tf.equal(flat_chars, Offsets.PAD), tf.int32), axis=1)
            with tf.control_dependencies([ech0]):
                embed_chars =  tf.nn.embedding_lookup(Wch, flat_chars)

            fwd_lstm = stacked_lstm(self.lstmsz // 2, self.pdrop, self.layers)
            bwd_lstm = stacked_lstm(self.lstmsz // 2, self.pdrop, self.layers)
            _, rnn_state = tf.nn.bidirectional_dynamic_rnn(fwd_lstm, bwd_lstm, embed_chars, sequence_length=word_lengths, dtype=tf.float32)

            result = tf.concat([rnn_state[0][-1].h, rnn_state[1][-1].h], axis=1)
            return tf.reshape(result, [B, T, self.lstmsz])

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.vsz

    def save_md(self, target):
        write_json({'vsz': self.vsz, 'dsz': self.dsz, 'lstmsz': self.lstmsz}, target)
