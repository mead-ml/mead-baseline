import math
import copy
import logging
import numpy as np
import tensorflow as tf
from baseline.utils import write_json, Offsets
from baseline.embeddings import register_embeddings
from baseline.tf.tfy import (char_word_conv_embeddings,
                             get_shape_as_list,
                             stacked_lstm,
                             highway_conns,
                             skip_conns,
                             tf_activation,
                             TRAIN_FLAG)
from baseline.utils import is_sequence


FLOAT32 = 4
GB2 = 1024 * 1024 * 1024 * 2
logger = logging.getLogger('baseline')


class TensorFlowEmbeddings(object):
    """This provides a base for TensorFlow embeddings sub-graphs

    """
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """Constructor
        """
        super(TensorFlowEmbeddings, self).__init__()
        self.name = name
        self.trainable = trainable
        self.dtype = dtype
        self._record_state(**kwargs)

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

    def call(self, x):
        return self.encode(x)

    def save_md(self):
        """Save the meta-data associated with this object, namely the `vsz` and `dsz`

        :return:
        """
        pass

    def get_feed_dict(self):
        """Return a feed dict that is needed to initialize this embeddings."""
        return {}

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
        # If we think we are going to hit the 2GB limit swap out the LUT
        # embeddings to use the placeholder trick to get around it.
        if cls is LookupTableEmbeddings and model.vsz * model.dsz * FLOAT32 > GB2:
            cls = LargeLookupTableEmbeddings
            logger.warning("Embedding %s seems to be larger than 2GB", name)
        return cls(name, vsz=model.vsz, dsz=model.dsz, weights=model.weights, **kwargs)

    def _record_state(self, **kwargs):
        w = kwargs.pop('weights', None)
        self._state = copy.deepcopy(kwargs)

    def save_md(self, target):
        """Save the metadata associated with this embedding as a JSON file

        :param target: The name of the output file
        :return:
        """
        write_json(self.get_config(), target)

    def get_config(self):
        #config = super(TensorFlowEmbeddings, self).get_config()
        config = {}
        config['module'] = self.__class__.__module__
        config['class'] = self.__class__.__name__
        config.update(self._state)
        return config


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
        super(LookupTableEmbeddings, self).__init__(name=name, **kwargs)

        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.scope = kwargs.get('scope', '{}/LUT'.format(self.name))
        self.dropin = kwargs.get('dropin', 0.0)
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
                                     dropin=self.dropin,
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
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            W = tf.get_variable("W",
                                initializer=tf.constant_initializer(self.weights, dtype=tf.float32, verify_shape=True),
                                shape=[self.vsz, self.dsz], trainable=self.finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))

            with tf.control_dependencies([e0]):
                embedding_w_dropout = tf.layers.dropout(W, self.dropin, noise_shape=(self.vsz, 1),  training=TRAIN_FLAG())
                word_embeddings = tf.nn.embedding_lookup(embedding_w_dropout, self.x)

        return word_embeddings


@register_embeddings(name='large-lut')
class LargeLookupTableEmbeddings(LookupTableEmbeddings):
    """Provide "classic" Lookup-Table based word embeddings

    """

    def encode(self, x=None):
        """Create a really large embedding matrix in tensorflow.

        Tensorflow has a limit on the size that a op can be (2GB). When we have very
        large embedding lookuptables (for example when we don't prune the vocab) we
        hit this limit and can't have the embeddings in the graph. This is due to a
        limit in the size that a thing can be in a protocol buffer (how tensorflow
        serializes the graph).

        Here we get around it with a placeholder. The place holder will be in the
        graph and it will know it needs to have a size of [vsz, dsz] but it doesn't
        have the actual values so it can be serialized into a protocol buffer since
        it is small.

        We then have a variable that is initialized with the value of the
        placeholder. This is filled in with value during the `sess.run` of
        `tf.global_variables_initialzier` with a feed_dict. Values are then saved
        into the checkpoint and can be reloaded from there.

        ```
        sess.run(tf.global_variables_initializer(), {e.W_place: e.weights})
        ```
        """
        if x is None:
            x = LookupTableEmbeddings.create_placeholder(self.name)
        self.x = x

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):

            self.W_place = tf.placeholder(tf.float32, shape=(self.vsz, self.dsz))
            W = tf.get_variable("W", initializer=self.W_place)

            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))

            with tf.control_dependencies([e0]):
                embedding_w_dropout = tf.layers.dropout(W, self.dropin, noise_shape=(self.vsz, 1),  training=TRAIN_FLAG())
                word_embeddings = tf.nn.embedding_lookup(embedding_w_dropout, self.x)

        return word_embeddings

    def get_feed_dict(self):
        """Feed dict mapping the numpy weights to the placeholder."""
        return {self.W_place: self.weights}


@register_embeddings(name='char-conv')
class CharConvEmbeddings(TensorFlowEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def _get_filtsz(self):
        # If this is a list, then its a tuple of (filtsz, nfeats)
        if is_sequence(self.cfiltsz[0]):
            filtsz = [filter_and_size[0] for filter_and_size in self.cfiltsz]
            nfeats = [filter_and_size[1] for filter_and_size in self.cfiltsz]

        # If we get a nfeat factor, we multiply that by each filter, and thresh at max_feat
        elif self.nfeat_factor:
            max_feat = self.max_feat
            filtsz = self.cfiltsz
            nfeats = [min(self.nfeat_factor * fsz, max_feat) for fsz in filtsz]
        # Otherwise its just a scalar
        else:
            nfeats = self.wsz
            filtsz = self.cfiltsz
        return filtsz, nfeats

    def __init__(self, name, **kwargs):
        super(CharConvEmbeddings, self).__init__(name=name, **kwargs)
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
        self.projsz = kwargs.get('projsz')
        self.dropin = kwargs.get('dropin', 0.0)
        self.x = None

        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

        with tf.device("/cpu:0"):
            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                self.Wch = tf.get_variable('Wch',
                                           initializer=tf.constant_initializer(self.weights, dtype=tf.float32,
                                                                               verify_shape=True),
                                           shape=[self.vsz, self.dsz], trainable=True)
        # These are the actual final filter sizes and num features
        self.filtsz, self.nfeats = self._get_filtsz()
        self.outsz = np.sum(self.nfeats)

        if self.projsz:
            with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
                self.Wp = tf.get_variable('Wp', shape=[self.outsz, self.projsz], trainable=True)
                self.bp = tf.get_variable('bp', shape=[self.projsz], trainable=True, initializer=tf.constant_initializer(0.0))
            self.outsz = self.projsz

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
                                  weights=self.weights,
                                  dropin=self.dropin,
                                  projsz=self.projsz)

    def encode(self, x=None):
        if x is None:
            x = CharConvEmbeddings.create_placeholder(self.name)
        self.x = x

        ech0 = tf.scatter_update(self.Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))

        mxlen = tf.shape(self.x)[1]

        gating_fn = highway_conns if self.gating.startswith('highway') else skip_conns

        with tf.variable_scope("Chars2Word"):
            with tf.control_dependencies([ech0]):
                mxwlen = tf.shape(self.x)[-1]
                char_bt_x_w = tf.reshape(self.x, [-1, mxwlen])
                # The ablation table (4) in https://arxiv.org/pdf/1708.02182.pdf shows this has a massive impact
                embedding_w_dropout = tf.layers.dropout(self.Wch, self.dropin, noise_shape=(self.vsz, 1), training=TRAIN_FLAG())
                cembed = tf.nn.embedding_lookup(embedding_w_dropout, char_bt_x_w, name="embeddings")
                cmot, num_filts = char_word_conv_embeddings(cembed, self.filtsz, self.dsz, self.nfeats,
                                                            activation_fn=tf_activation(self.activation),
                                                            gating=gating_fn,
                                                            num_gates=self.num_gates)

        if self.projsz:
            cmot = tf.matmul(cmot, self.Wp) + self.bp
        word_char = tf.reshape(cmot, [-1, mxlen, self.outsz])
        return word_char

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.outsz

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


@register_embeddings(name='positional-char-conv')
class PositionalCharConvEmbeddings(CharConvEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(PositionalCharConvEmbeddings, self).__init__(name=name, **kwargs)
        self.max_timescale = kwargs.get('max_timescale', 1.0e4)

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return PositionalCharConvEmbeddings(name=self.name, vsz=self.vsz, dsz=self.dsz, scope=self.scope,
                                  finetune=self.finetune, nfeat_factor=self.nfeat_factor,
                                  cfiltsz=self.cfiltsz, max_feat=self.max_feat, gating=self.gating,
                                  num_gates=self.num_gates, activation=self.activation, wsz=self.wsz,
                                  weights=self.weights)

    def encode(self, x=None):
        x = super(PositionalCharConvEmbeddings, self).encode(x) * math.sqrt(self.dsz)
        B, T, C = get_shape_as_list(x)
        signal = get_timing_signal_1d(T, C, min_timescale=1.0, max_timescale=self.max_timescale, start_index=0)
        return x + signal

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz


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


@register_embeddings(name='learned-positional')
class LearnedPositionalLookupTableEmbeddings(LookupTableEmbeddings):
    def __init__(self, name, **kwargs):
        super(LearnedPositionalLookupTableEmbeddings, self).__init__(name, **kwargs)
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_weights = kwargs.get('pos_weights')
        if self.pos_weights is None:
            unif = float(kwargs.get('unif', 0.1))
            self.pos_weights = np.random.uniform(-unif, unif, (self.mxlen, self.dsz))
        with tf.variable_scope(self.scope):
            self.pos = tf.get_variable("pos",
                                  initializer=tf.constant_initializer(self.pos_weights, dtype=tf.float32, verify_shape=True),
                                  shape=[self.mxlen, self.dsz], trainable=True)

    def encode(self, x=None):
        x = super(LearnedPositionalLookupTableEmbeddings, self).encode(x)
        T = tf.shape(x)[1]
        e0 = tf.scatter_update(self.pos, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
        with tf.control_dependencies([e0]):
            pos_embeddings = tf.nn.embedding_lookup(self.pos, tf.range(T, dtype=tf.int32))

        return x + tf.expand_dims(pos_embeddings, 0)

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return LearnedPositionalLookupTableEmbeddings(self.name,
                                                      vsz=self.vsz,
                                                      dsz=self.dsz,
                                                      finetune=self.finetune,
                                                      scope=self.scope,
                                                      weights=self.weights,
                                                      pos_weights=self.pos_weights,
                                                      mxlen=self.mxlen)


@register_embeddings(name='learned-positional-char-conv')
class LearnedPositionalCharConvEmbeddings(CharConvEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(LearnedPositionalCharConvEmbeddings, self).__init__(name, **kwargs)
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_weights = kwargs.get('pos_weights')
        if self.pos_weights is None:
            unif = float(kwargs.get('unif', 0.1))
            self.pos_weights = np.random.uniform(-unif, unif, (self.mxlen, self.dsz))
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.pos = tf.get_variable("pos",
                                       initializer=tf.constant_initializer(self.pos_weights, dtype=tf.float32, verify_shape=True),
                                       shape=[self.mxlen, self.dsz], trainable=True)

    def detached_ref(self):
        """This will detach any attached input and reference the same sub-graph otherwise

        :return:
        """
        if self.weights is None:
            raise Exception('You must initialize `weights` in order to use this method')
        return LearnedPositionalCharConvEmbeddings(name=self.name,
                                                   vsz=self.vsz,
                                                   dsz=self.dsz,
                                                   scope=self.scope,
                                                   finetune=self.finetune,
                                                   nfeat_factor=self.nfeat_factor,
                                                   cfiltsz=self.cfiltsz,
                                                   max_feat=self.max_feat,
                                                   gating=self.gating,
                                                   num_gates=self.num_gates,
                                                   activation=self.activation,
                                                   wsz=self.wsz,
                                                   weights=self.weights,
                                                   pos_weights=self.pos_weights,
                                                   mxlen=self.mxlen)

    def encode(self, x=None):
        x = super(LearnedPositionalCharConvEmbeddings, self).encode(x)
        T = tf.shape(x)[1]
        e0 = tf.scatter_update(self.pos, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))
        with tf.control_dependencies([e0]):
            pos_embeddings = tf.nn.embedding_lookup(self.pos, tf.range(T, dtype=tf.int32))

        return x + tf.expand_dims(pos_embeddings, 0)

    def _record_state(self, **kwargs):
        _ = kwargs.pop('pos_weights', None)
        super(LearnedPositionalCharConvEmbeddings, self)._record_state(**kwargs)

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.wsz




@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(TensorFlowEmbeddings):
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharLSTMEmbeddings, self).__init__(name=name, **kwargs)
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
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
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
