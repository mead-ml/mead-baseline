import math
import copy
import logging
import numpy as np
import tensorflow as tf
from eight_mile.utils import write_json, Offsets, is_sequence, calc_nfeats
from eight_mile.tf.layers import *


FLOAT32 = 4
GB2 = 1024 * 1024 * 1024 * 2
logger = logging.getLogger("baseline")


class TensorFlowEmbeddings(tf.keras.layers.Layer):
    """This provides a base for TensorFlow embeddings sub-graphs

    """

    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        # tf.kers.layers.Layer has a validation step that only allows certain kwargs
        # to be passed into it. These are not documented and you need to look into the
        # code to find this. For now just don't pass in out kwargs
        super().__init__(trainable=trainable, name=name, dtype=dtype)
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

    def encode(self, *x):
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
    def __init__(self, trainable=True, name=None, dtype=tf.float32, cpu_placement=False, **kwargs):
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
        trainable = kwargs.get("finetune", trainable)
        # The layers have a filter of allowed keywords and the docs don't list what they are
        # you need to look in code. We are just not passing kwargs for now.
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.vsz = kwargs.get("vsz")
        self.padding_idx = kwargs.get('padding_idx', Offsets.PAD)
        self.dsz = kwargs.get("dsz")
        self.finetune = kwargs.get("finetune", trainable)
        self.scope = kwargs.get("scope", "LUT")
        self.dropin = kwargs.get("dropin", 0.0)
        self._weights = kwargs.get("weights")
        self.drop = tf.keras.layers.Dropout(rate=self.dropin, noise_shape=(self.get_vsz(), 1))
        self.cpu_placement = cpu_placement
        if self._weights is None:
            unif = kwargs.get("unif", 0.1)
            self._weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        else:
            self.vsz, self.dsz = self._weights.shape

    def build(self, input_shape):

        if self.cpu_placement:
            with tf.device("cpu:0"):
                self.W = self.add_weight(
                        name=f"{self.scope}/Weight",
                        shape=(self.vsz, self.dsz),
                        initializer=tf.constant_initializer(self._weights),
                        trainable=self.finetune,
                )
        else:
            self.W = self.add_weight(
                name=f"{self.scope}/Weight",
                shape=(self.vsz, self.dsz),
                initializer=tf.constant_initializer(self._weights),
                trainable=self.finetune,
            )
        super().build(input_shape)

    def _embed_w_dropout(self, x):
        # The ablation table (4) in https://arxiv.org/pdf/1708.02182.pdf shows this has a massive impact
        embedding_w_dropout = self.drop(self.W, training=TRAIN_FLAG())
        word_embeddings = tf.nn.embedding_lookup(embedding_w_dropout, x)
        return word_embeddings

    def encode(self, *x):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        self.x = x[0]
        if self.padding_idx is not None:
            e0 = tf.tensor_scatter_nd_update(
                self.W, tf.constant(self.padding_idx, dtype=tf.int32, shape=[1, 1]), tf.zeros(shape=[1, self.dsz])
            )
            with tf.control_dependencies([e0]):
                return self._embed_w_dropout(self.x)
        else:
            return self._embed_w_dropout(self.x)

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
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.cpu_placement = bool(kwargs.get('cpu_placement', False))
        self.scope = kwargs.get("scope", "CharConv")
        self.finetune = kwargs.get("finetune", trainable)
        self.nfeat_factor = kwargs.get("nfeat_factor", None)
        self.cfiltsz = kwargs.get("cfiltsz", kwargs.get("filtsz", [3]))
        self.max_feat = kwargs.get("max_feat", 30)
        gating = kwargs.get("gating", "skip")
        num_gates = kwargs.get("num_gates", 1)
        activation = kwargs.get("activation", "tanh")
        self.wsz = kwargs.get("wsz", 30)
        self.projsz = kwargs.get("projsz")
        self.x = None
        # These are the actual final filter sizes and num features
        filtsz, nfeats = calc_nfeats(self.cfiltsz, self.nfeat_factor, self.max_feat, self.wsz)

        self.embed = LookupTableEmbeddings(name=f"{self.name}/CharLUT", finetune=self.finetune, **kwargs)
        dsz = self.embed.output_dim
        self.parallel_conv = ParallelConv(dsz, nfeats, filtsz, activation)
        self.gating_fns = tf.keras.Sequential()
        for _ in range(num_gates):
            if gating == 'skip':
                self.gating_fns.add(SkipConnection(self.parallel_conv.output_dim, activation))
            else:
                self.gating_fns.add(Highway(self.parallel_conv.output_dim))

        self.outsz = self.parallel_conv.output_dim
        if self.projsz is not None:
            self.outsz = self.projsz
            self.proj = tf.keras.layers.Dense(self.outsz, bias_initializer=tf.constant_initializer(0.0))



    @property
    def dsz(self):
        return self.outsz

    def encode(self, *x):
        self.x = x[0]

        mxlen = tf.shape(self.x)[1]
        mxwlen = tf.shape(self.x)[-1]
        char_bt_x_w = tf.reshape(self.x, [-1, mxwlen])
        cembed = self.embed(char_bt_x_w)
        cmot = self.parallel_conv(cembed)
        cmot = self.gating_fns(cmot)
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
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.scope = kwargs.get("scope", "CharLUT")
        self.finetune = kwargs.get("finetune", trainable)
        self.lstmsz = kwargs.get("lstmsz", 50)
        self.lstm_layers = kwargs.get("layers", 1)
        self.pdrop = kwargs.get("pdrop", 0.5)
        self.rnn_type = kwargs.get("rnn_type", "blstm")
        self.x = None
        self.embed = LookupTableEmbeddings(name=f"{self.name}/CharLUT", finetune=self.finetune, **kwargs)
        self.lstm = BiLSTMEncoderHidden(
            self.embed.output_dim,
            self.lstmsz,
            self.lstm_layers,
            pdrop=self.pdrop,
            requires_length=True,
            name=f"{self.name}/blstm",
        )

    def encode(self, *x):
        self.x = x[0]
        shape = tf.shape(self.x)
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
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.scope = kwargs.get("scope", "CharLUT")
        self.finetune = kwargs.get("finetune", trainable)
        self.embed = LookupTableEmbeddings(name=f"{self.name}/CharLUT", finetune=self.finetune, **kwargs)
        self.d_model = kwargs.get("wsz", 30)
        self.num_heads = kwargs.get("num_heads", 3)
        self.rpr_k = kwargs.get("rpr_k", 10)
        layers = kwargs.get("layers", 1)
        pdrop = kwargs.get("pdrop", 0.5)
        self.char_comp = TransformerEncoderStackWithLengths(
            self.num_heads,
            self.d_model,
            pdrop,
            False,
            layers,
            rpr_k=self.rpr_k,
            input_sz=self.embed.output_dim,
            name=f"{self.name}/transformer",
        )

    def encode(self, *x):
        self.x = x[0]
        shape = tf.shape(self.x)
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


class PositionalMixin(tf.keras.layers.Layer):
    def positional(self, length):
        pass


class SinusoidalPositionalMixin(PositionalMixin):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)

        max_timescale = kwargs.get("max_timescale", 1.0e4)
        # Match the mxlen pytorch has because it precomputes the timing signal
        mxlen = kwargs.get('mxlen', 10000)

        word_dsz = self.get_dsz()
        log_timescale_increment = math.log(max_timescale) / float(word_dsz)
        inv_timescales = np.exp(np.arange(0, word_dsz, 2, dtype=np.float32) * -log_timescale_increment)

        pe = np.zeros((mxlen, word_dsz), dtype=np.float32)
        position = np.expand_dims(np.arange(0, mxlen, dtype=np.float32), 1)

        pe[:, 0::2] = np.sin(position * inv_timescales)
        pe[:, 1::2] = np.cos(position * inv_timescales)

        self.pe = tf.expand_dims(pe, 0)

    def positional(self, length):
        return self.pe[:, :length]


class LearnedPositionalMixin(PositionalMixin):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.mxlen = int(kwargs.get("mxlen", 512))
        self.pos_weights = kwargs.get("pos_weights")
        if self.pos_weights is None:
            unif = float(kwargs.get("unif", 0.1))
            self.pos_weights = np.random.uniform(-unif, unif, (self.mxlen, self.get_dsz()))

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos",
            initializer=tf.constant_initializer(self.pos_weights),
            shape=[self.mxlen, self.get_dsz()],
            trainable=self.finetune,
        )
        super().build(input_shape)

    def positional(self, length):
        return tf.expand_dims(tf.nn.embedding_lookup(self.pos, tf.range(length, dtype=tf.int32)), 0)


class PositionalLookupTableEmbeddings(SinusoidalPositionalMixin, LookupTableEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x) * tf.constant(self.scale)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())


class LearnedPositionalLookupTableEmbeddings(LearnedPositionalMixin, LookupTableEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())


class LearnedPositionalLookupTableEmbeddingsWithBias(LearnedPositionalMixin, LookupTableEmbeddings):
    """Learned positional lookup table embeddings wih a bias and layer norm

    This is just a typical learned positional embedding but with a learnable
    bias and a layer norm.  This is equivalent to BERT embeddings when the
    token_type is not set

    """
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.bias = self.add_weight(
            name="bias",
            initializer=tf.constant_initializer(0.0),
            shape=[1, self.get_dsz()],
            trainable=self.finetune,
        )

    def encode(self, *x):
        y = super().encode(*x)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        y = y + pos + self.bias
        return y


class PositionalCharConvEmbeddings(SinusoidalPositionalMixin, CharConvEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x) * tf.constant(self.scale)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())


class LearnedPositionalCharConvEmbeddings(LearnedPositionalMixin, CharConvEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())


class PositionalCharLSTMEmbeddings(SinusoidalPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.scale = math.sqrt(self.get_dsz())
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x) * tf.constant(self.scale)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())


class LearnedPositionalCharLSTMEmbeddings(LearnedPositionalMixin, CharLSTMEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.dropout = tf.keras.layers.Dropout(kwargs.get("dropout", 0.0))

    def encode(self, *x):
        y = super().encode(*x)
        T = tf.shape(y)[1]
        pos = self.positional(T)
        return self.dropout(y + pos, training=TRAIN_FLAG())
