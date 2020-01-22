import math
import copy
import logging
import numpy as np
import tensorflow as tf
from eight_mile.utils import write_json, Offsets, is_sequence, calc_nfeats
from eight_mile.embeddings import register_embeddings
from eight_mile.tf.layers import *


FLOAT32 = 4
GB2 = 1024 * 1024 * 1024 * 2
logger = logging.getLogger('baseline')


class TensorFlowEmbeddings(tf.keras.layers.Layer):
    """This provides a base for TensorFlow embeddings sub-graphs

    """
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        # tf.kers.layers.Layer has a validation step that only allows certain kwargs
        # to be passed into it. These are not documented and you need to look into the
        # code to find this. For now just don't pass in out kwargs
        super().__init__(trainable, name, dtype)
        self._name = name
        self.W = None

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

    def get_weights(self):
        raise NotImplementedError

    def encode(self, x):
        """This defines the computation of the sub-graph for this object and returns the output node

        :return:
        """
        pass

    @property
    def output_dim(self):
        return self.get_dsz()

    def call(self, x):
        return self.encode(x)

    def get_feed_dict(self):
        """Return a feed dict that is needed to initialize this embeddings."""
        return {}


class LookupTableEmbeddings(TensorFlowEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        """Create a lookup-table based embedding.

        :param name: The name of the feature/placeholder, and a key for the scope
        :param kwargs:

        :Keyword Arguments: See below
        * *vsz* -- (``int``) this is the vocabulary (input) size of the lookup table
        * *dsz* -- (``int``) the output dimension size of this embedding
        * *finetune* -- (``bool``) (default is `True`) should we allow the sub-graph to learn updated weights
        * *weights* -- (``numpy.ndarray``) Optional `vsz x dsz` weight matrix for initialization
        * *unif* -- (``float``) (defaults to `0.1`) If the weights should be created, what is the random initialization range
        """
        trainable = kwargs.get('finetune', trainable)
        # The layers have a filter of allowed keywords and the docs don't list what they are
        # you need to look in code. We are just not passing kwargs for now.
        super().__init__(trainable=trainable, name=name, dtype=dtype)

        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', trainable)
        self.scope = kwargs.get("scope", "LUT")
        self.dropin = kwargs.get('dropin', 0.0)
        self._weights = kwargs.get('weights')
        self.drop = tf.keras.layers.Dropout(rate=self.dropin, noise_shape=(self.get_vsz(), 1))

        if self._weights is None:
            unif = kwargs.get('unif', 0.1)
            self._weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        else:
            self.vsz, self.dsz = self._weights.shape

    def build(self, input_shape):

        if tf.executing_eagerly():
            with tf.device('cpu:0'):
                self.W = self.add_weight(
                    name=f'{self.scope}/Weight',
                    shape=(self.vsz, self.dsz),
                    initializer=tf.constant_initializer(
                        self._weights,
                    ),
                    trainable=self.finetune
                )
        else:
            self.W = self.add_weight(
                name=f'{self.scope}/Weight',
                shape=(self.vsz, self.dsz),
                initializer=tf.constant_initializer(
                    self._weights,
                ),
                trainable=self.finetune
            )
        super().build(input_shape)

    def encode(self, x):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        self.x = x
        e0 = tf.tensor_scatter_nd_update(
            self.W,
            tf.constant(Offsets.PAD, dtype=tf.int32, shape=[1, 1]),
            tf.zeros(shape=[1, self.dsz])
        )
        with tf.control_dependencies([e0]):
            # The ablation table (4) in https://arxiv.org/pdf/1708.02182.pdf shows this has a massive impact
            embedding_w_dropout = self.drop(self.W, training=TRAIN_FLAG())
            word_embeddings = tf.nn.embedding_lookup(embedding_w_dropout, self.x)

        return word_embeddings

    def get_vsz(self):
        return self.vsz

    def get_dsz(self):
        return self.dsz

    def get_weights(self):
        return self.W


class CharConvEmbeddings(TensorFlowEmbeddings):
    """dos Santos embeddings extended to parallel filters (AKA Kim character-aware neural language model inputs)

    """

    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self._name = name
        self.scope = kwargs.get('scope', 'CharConv')
        self.finetune = kwargs.get('finetune', trainable)
        self.nfeat_factor = kwargs.get('nfeat_factor', None)
        self.cfiltsz = kwargs.get('cfiltsz', kwargs.get('filtsz', [3]))
        self.max_feat = kwargs.get('max_feat', 30)
        self.gating = kwargs.get('gating', 'skip')
        self.num_gates = kwargs.get('num_gates', 1)
        self.activation = kwargs.get('activation', 'tanh')
        self.wsz = kwargs.get('wsz', 30)
        self.projsz = kwargs.get('projsz')
        self.x = None
        # These are the actual final filter sizes and num features
        self.filtsz, self.nfeats = calc_nfeats(self.cfiltsz, self.nfeat_factor, self.max_feat, self.wsz)
        self.conv_outsz = np.sum(self.nfeats)
        self.outsz = self.conv_outsz
        if self.projsz is not None:
            self.outsz = self.projsz
            self.proj = tf.keras.layers.Dense(self.outsz, bias_initializer=tf.constant_initializer(0.0))

        self.embed = LookupTableEmbeddings(name=f'{self.name}/CharLUT', finetune=self.finetune, **kwargs)

    def encode(self, x):
        self.x = x

        mxlen = tf.shape(self.x)[1]
        gating_fn = highway_conns if self.gating.startswith('highway') else skip_conns

        mxwlen = tf.shape(self.x)[-1]
        char_bt_x_w = tf.reshape(self.x, [-1, mxwlen])
        cembed = self.embed(char_bt_x_w)
        cmot, num_filts = char_word_conv_embeddings(
            cembed,
            self.filtsz,
            self.embed.output_dim,
            self.nfeats,
            activation_fn=get_activation(self.activation),
            gating=gating_fn,
            num_gates=self.num_gates
        )

        if self.projsz:
            cmot = self.proj(cmot)
        word_char = tf.reshape(cmot, [-1, mxlen, self.outsz])
        return word_char

    def get_vsz(self):
        return self.embed.get_vsz()

    def get_dsz(self):
        return self.outsz


class CharLSTMEmbeddings(TensorFlowEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.scope = kwargs.get('scope', 'CharLUT')
        self.finetune = kwargs.get('finetune', trainable)
        self.lstmsz = kwargs.get('lstmsz', 50)
        self.lstm_layers = kwargs.get('layers', 1)
        self.pdrop = kwargs.get('pdrop', 0.5)
        self.rnn_type = kwargs.get('rnn_type', 'blstm')
        self.x = None
        self.embed = LookupTableEmbeddings(name=f'{self.name}/CharLUT', finetune=self.finetune, **kwargs)
        self.lstm = BiLSTMEncoderHidden(self.embed.output_dim, self.lstmsz, self.lstm_layers, pdrop=self.pdrop, requires_length=True, name=f"{self.name}/blstm")

    def encode(self, x):
        self.x = x
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        W = shape[2]
        flat_chars = tf.reshape(x, [-1, W])
        embed_chars = self.embed(flat_chars)

        # Calculate the lengths of each word
        word_lengths = tf.reduce_sum(tf.cast(tf.not_equal(flat_chars, Offsets.PAD), tf.int32), axis=1)

        # cuDNN throws an error if there is an input with a length of 0, this happens when the "word"
        # is actually a "<PAD>" so there are no characters to run the LSTM over. Here we just say
        # that the lengths is 1. This will make cudnn happy and we will just get junk in that spot
        patched_lengths = tf.math.maximum(word_lengths, 1)

        # Run the LSTM
        result = self.lstm((embed_chars, patched_lengths))

        # Create a mask that is true when the length is 0 (where the word was a pad) so that
        # we can mask out the junk that the lstm created because we needed a length of 1
        result = tf.multiply(result, tf.expand_dims(tf.cast(tf.not_equal(word_lengths, 0), tf.float32), -1))

        return tf.reshape(result, (B, T, self.lstmsz))

    def call(self, inputs):
        return self.encode(inputs)

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.embed.get_vsz()


class CharTransformerEmbeddings(TensorFlowEmbeddings):

    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.scope = kwargs.get('scope', 'CharLUT')
        self.finetune = kwargs.get('finetune', trainable)
        self.embed = LookupTableEmbeddings(name=f'{self.name}/CharLUT', finetune=self.finetune, **kwargs)
        self.d_model = kwargs.get('wsz', 30)
        self.num_heads = kwargs.get('num_heads', 3)
        self.rpr_k = kwargs.get('rpr_k', 10)
        layers = kwargs.get('layers', 1)
        pdrop = kwargs.get('pdrop', 0.5)
        self.char_comp = TransformerEncoderStackWithLengths(self.num_heads, self.d_model, pdrop, False, layers,
                                                            rpr_k=self.rpr_k, input_sz=self.embed.output_dim,
                                                            name=f"{self.name}/transformer")

    def encode(self, x):
        self.x = x
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        W = shape[2]
        flat_chars = tf.reshape(x, [-1, W])
        embed_chars = self.embed(flat_chars)

        # Calculate the lengths of each word
        lengths = tf.reduce_sum(tf.cast(tf.not_equal(flat_chars, Offsets.PAD), tf.int32), axis=1)

        # Run the LSTM
        result = self.char_comp((embed_chars, lengths))

        pooled = tf.reduce_max(result, -2, keepdims=False)

        return tf.reshape(pooled, (B, T, self.d_model))

    def call(self, inputs):
        return self.encode(inputs)

    def get_dsz(self):
        return self.d_model

    def get_vsz(self):
        return self.embed.get_vsz()


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
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            tf.maximum(tf.cast(num_timescales, tf.float32) - 1, 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, channels % 2]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


class PositionalMixin(tf.keras.layers.Layer):
    def positional(self, length):
        pass


class SinusoidalPositionalMixin(PositionalMixin):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.max_timescale = kwargs.get('max_timescale', 1.0e4)
        # Match the mxlen pytorch has because it precomputes the timing signal
        self.mxlen = 10000
        self.min_timescale = kwargs.get('min_timescale', 1.0)

    def positional(self, length):
        return get_timing_signal_1d(length, self.get_dsz(), min_timescale=self.min_timescale, max_timescale=self.max_timescale, start_index=0)


class LearnedPositionalMixin(PositionalMixin):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.mxlen = int(kwargs.get('mxlen', 512))
        self.pos_weights = kwargs.get('pos_weights')
        if self.pos_weights is None:
            unif = float(kwargs.get('unif', 0.1))
            self.pos_weights = np.random.uniform(-unif, unif, (self.mxlen, self.get_dsz()))

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos",
            initializer=tf.constant_initializer(self.pos_weights),
            shape=[self.mxlen, self.get_dsz()],
            trainable=self.finetune
        )
        super().build(input_shape)

    def positional(self, length):
        return tf.expand_dims(tf.nn.embedding_lookup(self.pos, tf.range(length, dtype=tf.int32)), 0)


class PositionalLookupTableEmbeddings(SinusoidalPositionalMixin, LookupTableEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(name=name, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x) * tf.constant(self.scale)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


class LearnedPositionalLookupTableEmbeddings(LearnedPositionalMixin, LookupTableEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


class PositionalCharConvEmbeddings(SinusoidalPositionalMixin, CharConvEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(name=name, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x) * tf.constant(self.scale)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


class LearnedPositionalCharConvEmbeddings(LearnedPositionalMixin, CharConvEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


class PositionalCharLSTMEmbeddings(SinusoidalPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x) * tf.constant(self.scale)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


class LearnedPositionalCharLSTMEmbeddings(LearnedPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainalbe=trainable, name=name, dtype=dtype, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get('dropout', 0.1))

    def encode(self, x):
        x = super().encode(x)
        T = tf.shape(x)[1]
        pos = self.positional(T)
        return self.dropout(x + pos, training=TRAIN_FLAG())


# All the train functions don't have the large lut init codes anymore and it needs a placeholder
# So lets skip it for now
# https://stackoverflow.com/questions/43288147/how-do-i-use-a-very-large-2m-word-embedding-in-tensorflow
# class LargeLookupTableEmbeddings(LookupTableEmbeddings):
#     """Create a really large embedding matrix in tensorflow.

#     Tensorflow has a limit on the size that a op can be (2GB). When we have very
#     large embedding lookuptables (for example when we don't prune the vocab) we
#     hit this limit and can't have the embeddings in the graph. This is due to a
#     limit in the size that a thing can be in a protocol buffer (how tensorflow
#     serializes the graph).

#     Here we get around it with a placeholder. The place holder will be in the
#     graph and it will know it needs to have a size of [vsz, dsz] but it doesn't
#     have the actual values so it can be serialized into a protocol buffer since
#     it is small.

#     We then have a variable that is initialized with the value of the
#     placeholder. This is filled in with value during the `sess.run` of
#     `tf.global_variables_initialzier` with a feed_dict. Values are then saved
#     into the checkpoint and can be reloaded from there.

#     ```
#     sess.run(tf.global_variables_initializer(), {e.W_place: e._weights})
#     ```

#     The future of this with tf 2 is unclear because it uses a placeholder to
#     get around the size limit and placeholders are not cool in tf2
#     """
#     def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
#         super().__init__(trainable, name, dtype, **kwargs)
#         self.W_place = tf.placeholder(tf.float32, shape=(self.vsz, self.dsz))

#     def build(self, input_shape):
#         self.W = tf.get_variable(f"{self.scope}/weight", initializer=self.W_place)

#     def get_feed_dict(self):
#         """Feed dict mapping the numpy weights to the placeholder."""
#         return {self.W_place: self._weights}
