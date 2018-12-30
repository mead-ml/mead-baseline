import tensorflow as tf
import numpy as np
from baseline.utils import listify, Offsets
import math
BASELINE_TF_TRAIN_FLAG = None


def SET_TRAIN_FLAG(X):
    global BASELINE_TF_TRAIN_FLAG
    BASELINE_TF_TRAIN_FLAG = X


def TRAIN_FLAG():
    """Create a global training flag on first use"""
    global BASELINE_TF_TRAIN_FLAG
    if BASELINE_TF_TRAIN_FLAG is not None:
        return BASELINE_TF_TRAIN_FLAG

    BASELINE_TF_TRAIN_FLAG = tf.placeholder_with_default(False, shape=(), name="TRAIN_FLAG")
    return BASELINE_TF_TRAIN_FLAG


def new_placeholder_dict(train):
    global BASELINE_TF_TRAIN_FLAG

    if train:
        return {BASELINE_TF_TRAIN_FLAG: 1}
    return {}


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))


def swish(x):
    return x*tf.nn.sigmoid(x)


def get_shape_as_list(x):
    """
    This function makes sure we get a number whenever possible, and otherwise, gives us back
    a graph operation, but in both cases, presents as a list.  This makes it suitable for a
    bunch of different operations within TF, and hides away some details that we really dont care about, but are
    a PITA to get right...

    Borrowed from Alec Radford:
    https://github.com/openai/finetune-transformer-lm/blob/master/utils.py#L38
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def tf_activation(name):
    if name == 'softmax':
        return tf.nn.softmax
    if name == 'tanh':
        return tf.nn.tanh
    if name == 'sigmoid':
        return tf.nn.sigmoid
    if name == 'gelu':
        return gelu
    if name == 'swish':
        return swish
    if name == 'ident':
        return tf.identity
    if name == 'leaky_relu':
        return tf.nn.leaky_relu
    return tf.nn.relu


class ParallelConvEncoder(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(self, dsz, motsz, filtsz, activation='relu', name=None, **kwargs):

        super(ParallelConv, self).__init__(name=name)
        self.Ws = []
        self.bs = []
        self.activation = tf_activation(activation)

        if not isinstance(motsz, list):
            motsz = [motsz] * len(filtsz)

        for fsz, cmotsz in zip(filtsz, motsz):
            kernel_shape = [1, int(fsz), int(dsz), int(cmotsz)]
            self.Ws.append(self.add_variable('cmot-{}/W'.format(fsz), shape=kernel_shape))
            self.bs.append(self.add_variable('cmot-{}/b'.format(fsz), shape=[cmotsz], initializer=tf.constant_initializer(0.0)))

        self.output_dim = sum(motsz)

    def call(self, inputs):

        parallels = []
        expanded = tf.expand_dims(inputs, ParallelConv.DUMMY_AXIS)
        for W, b in zip(self.Ws, self.bs):
            conv = tf.nn.conv2d(
                expanded, W,
                strides=[1, 1, 1, 1],
                padding="SAME", name="CONV"
            )
            activation = self.activation(tf.nn.bias_add(conv, b), 'activation')
            parallels.append(activation)
        combine = tf.reshape(tf.concat(values=activation, axis=ParallelConv.FEATURE_AXIS), [-1, self.output_dim])
        return combine

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [self.output_dim]

    @property
    def requires_length(self):
        return False


class ParallelConv(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(self, dsz, motsz, filtsz, activation='relu', name=None, **kwargs):
        """Do parallel convolutions with multiple filter widths and max-over-time pooling.

        :param filtsz: The list of filter widths to use.
        :param dsz: The depths of the input (H).
        :param motsz: The number of conv filters to use (can be an int or a list to allow for various sized filters)
        :param activation: (``str``) The name of the activation function to use (`default='relu`)
        """
        super(ParallelConv, self).__init__(name=name)
        self.Ws = []
        self.bs = []
        self.activation = tf_activation(activation)

        if not isinstance(motsz, list):
            motsz = [motsz] * len(filtsz)

        for fsz, cmotsz in zip(filtsz, motsz):
            kernel_shape = [1, int(fsz), int(dsz), int(cmotsz)]
            self.Ws.append(self.add_variable('cmot-{}/W'.format(fsz), shape=kernel_shape))
            self.bs.append(self.add_variable('cmot-{}/b'.format(fsz), shape=[cmotsz], initializer=tf.constant_initializer(0.0)))

        self.output_dim = sum(motsz)

    def call(self, inputs):
        """
        :param inputs: The inputs in the shape [B, T, H].
        :return: Combined result
        """
        mots = []
        expanded = tf.expand_dims(inputs, ParallelConv.DUMMY_AXIS)
        for W, b in zip(self.Ws, self.bs):
            conv = tf.nn.conv2d(
                expanded, W,
                strides=[1, 1, 1, 1],
                padding="SAME", name="CONV"
            )
            activation = self.activation(tf.nn.bias_add(conv, b), 'activation')
            mot = tf.reduce_max(activation, [ParallelConv.TIME_AXIS], keepdims=True)
            mots.append(mot)
        combine = tf.reshape(tf.concat(values=mots, axis=ParallelConv.FEATURE_AXIS), [-1, self.output_dim])
        return combine

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    @property
    def requires_length(self):
        return False


def tensor_and_lengths(inputs):
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None  ##tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), axis=1)

    return in_tensor, lengths


def rnn_ident(output, hidden):
    return output, hidden


def rnn_signal(output, hidden):
    return output


def lstm_cell(hsz, forget_bias=1.0):
    """Produce a single cell with no dropout

    :param hsz: (``int``) The number of hidden units per LSTM
    :param forget_bias: (``int``) Defaults to 1
    :return: a cell
    """
    return tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True)


def lstm_cell_w_dropout(hsz, pdrop, forget_bias=1.0, variational=False, training=False):
    """Produce a single cell with dropout

    :param hsz: (``int``) The number of hidden units per LSTM
    :param pdrop: (``int``) The probability of keeping a unit value during dropout
    :param forget_bias: (``int``) Defaults to 1
    :param variational (``bool``) variational recurrence is on
    :param training (``bool``) are we training? (defaults to ``False``)
    :return: a cell
    """
    output_keep_prob = tf.contrib.framework.smart_cond(training, lambda: 1.0 - pdrop, lambda: 1.0)
    state_keep_prob = tf.contrib.framework.smart_cond(training, lambda: 1.0 - pdrop if variational else 1.0, lambda: 1.0)
    cell = tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True)
    output = tf.contrib.rnn.DropoutWrapper(cell,
                                           output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob,
                                           variational_recurrent=variational,
                                           dtype=tf.float32)
    return output


def rnn_cell(hsz, rnntype, st=None):
    """Produce a single RNN cell

    :param hsz: (``int``) The number of hidden units per LSTM
    :param rnntype: (``str``): `lstm` or `gru`
    :param st: (``bool``) state is tuple? defaults to `None`
    :return: a cell
    """
    if st is not None:
        cell = tf.contrib.rnn.LSTMCell(hsz, state_is_tuple=st) if rnntype.endswith('lstm') else tf.contrib.rnn.GRUCell(hsz)
    else:
        cell = tf.contrib.rnn.LSTMCell(hsz) if rnntype.endswith('lstm') else tf.contrib.rnn.GRUCell(hsz)
    return cell


class StackedParallelConvEncoder(tf.keras.Model):

    def __init__(self, dsz, hsz, pdrop, layers, filts=[5], activation='relu', name=None, **kwargs):
        """Produce a stack of parallel or single convolution layers with residual connections and dropout between each

        :param hsz: (``int``) The number of hidden units per filter
        :param pdrop: (``float``) The probability of dropout
        :param layers: (``int``) The number of layers of parallel convolutions to stack
        :param filts: (``list``) A list of parallel filter widths to apply
        :param activation: (``str``) A name for activation
        :param name: A string name to scope this operation
        """
        super(StackedParallelConvEncoder, self).__init__(name=name)
        filts = listify(filts)
        self.layer_1 = tf.keras.Sequential([ParallelConvEncoder(dsz, hsz, filts, activation), tf.keras.layers.Dropout(pdrop)])

        self.subsequent = []
        for i in range(layers):
            new_block = tf.keras.Sequential([ParallelConvEncoder(hsz, hsz, filts, activation), tf.keras.layers.Dropout(pdrop)])
            self.subsequent.append(ResidualBlock(new_block))

    def call(self, inputs, training=False):
        """

        :param inputs: The input
        :param training:
        :return: a stacked CNN
        """
        x = self.layer_1(inputs, training)
        for layer in self.subsequent:
            x = layer(x)
        return layer


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, name=None, **kwargs):
        super(LayerNorm, self).__init__(name=name)
        self.axis = listify(axis)
        self.epsilon = epsilon

    def build(self, input_shape):
        n_state = input_shape[-1]
        self.gv = self.add_variable("g", [n_state], initializer=tf.constant_initializer(1))
        self.bv = self.add_variable("b", [n_state], initializer=tf.constant_initializer(0))
        super(LayerNorm, self).build(input_shape)

    def call(self, x, mask=None):
        u = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=self.axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + self.epsilon)
        x = x * self.gv + self.bv
        return x


class LSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, pdrop, nlayers, variational=False, output_fn=None, requires_length=True, name=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param variational (``bool``) variational recurrence is on
        :param training (``bool``) Are we training? (defaults to ``False``)
        :return: a stacked cell
        """
        super(LSTMEncoder, self).__init__(name=name)
        self._requires_length = requires_length

        if variational:
            self.rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=TRAIN_FLAG()) for _ in
                     range(nlayers)],
                    state_is_tuple=True
                )
        self.rnn = tf.contrib.rnn.MultiRNNCell(
            [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG()) if i < nlayers - 1 else lstm_cell(hsz) for i in range(nlayers)],
                state_is_tuple=True
        )
        self.output_fn = rnn_ident if output_fn is None else output_fn

    def call(self, inputs, training=False):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length


class BiLSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, pdrop, nlayers, variational=False, output_fn=None, requires_length=True, name=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param variational (``bool``) variational recurrence is on
        :param training (``bool``) Are we training? (defaults to ``False``)
        :return: a stacked cell
        """
        super(BiLSTMEncoder, self).__init__(name=name)
        self._requires_length = requires_length
        if variational:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=TRAIN_FLAG()) for _ in
                     range(nlayers)],
                    state_is_tuple=True
                )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=TRAIN_FLAG()) for _ in
                 range(nlayers)],
                state_is_tuple=True
                )
        else:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG()) if i < nlayers - 1 else lstm_cell(hsz) for i in range(nlayers)],
                    state_is_tuple=True
            )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG()) if i < nlayers - 1 else lstm_cell(hsz) for i in
                 range(nlayers)],
                state_is_tuple=True
            )
        self.output_fn = rnn_ident if output_fn is None else output_fn
        print(self.output_fn)

    def call(self, inputs, training=False):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.bidirectional_dynamic_rnn(self.fwd_rnn, self.bwd_rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        rnnout = tf.concat(axis=2, values=rnnout)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length


class EmbeddingsStack(tf.keras.Model):

    def __init__(self, embeddings_dict, requires_length=False, name=None, **kwargs):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """

        super(EmbeddingsStack, self).__init__(name=name)
        self.embeddings = embeddings_dict
        self._requires_length = requires_length

    def call(self, inputs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        for k, embedding in self.embeddings.items():
            x = inputs[k]
            embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
        word_embeddings = tf.concat(values=all_embeddings_out, axis=-1)
        return word_embeddings

    @property
    def requires_length(self):
        return self.requires_length


class DenseStack(tf.keras.Model):

    def __init__(self, hsz, activation='relu', pdrop_value=0.5, init=None, name=None, **kwargs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param hsz: (``int``) The number of hidden units
        :param activation:  (``str``) The name of the activation function to use
        :param pdrop_value: (``float``) The dropout probability
        :param init: The tensorflow initializer

        """
        super(DenseStack, self).__init__(name=name)
        hszs = listify(hsz)
        self.layer_stack = [tf.keras.layers.Dense(hsz, kernel_initializer=init, activation=activation) for hsz in hszs]
        self.dropout = tf.keras.layers.Dropout(pdrop_value)

    def call(self, inputs, training=False):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param inputs: The fixed representation of the model
        :param training: (``bool``) A boolean specifying if we are training or not
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """
        x = inputs
        for layer in self.layer_stack:
            x = layer(x)
            x = self.dropout(x, training)
        return x

    @property
    def requires_length(self):
        return False


class Highway(tf.keras.Model):

    def __init__(self, input_size, name=None, **kwargs):
        super(Highway, self).__init__(name=name)
        self.proj = tf.keras.layers.Dense(input_size, activation='relu')
        self.transform = tf.keras.layers.Dense(input_size, bias_initializer=tf.keras.initializers.Constant(value=-2.0), activation='sigmoid')

    def call(self, inputs):
        proj_result = self.proj(inputs)
        proj_gate = self.transform(inputs)
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * inputs)
        return gated

    @property
    def requires_length(self):
        return False


class ResidualBlock(tf.keras.Model):

    def __init__(self, layer=None, name=None, **kwargs):
        super(ResidualBlock, self).__init__(name=name)
        self.layer = layer

    def call(self, inputs):
        return inputs + self.layer(inputs)

    @property
    def requires_length(self):
        return False


class SkipConnection(ResidualBlock):

    def __init__(self, input_size, activation='relu'):
        super(SkipConnection, self).__init__(tf.keras.layers.Dense(input_size, activation=activation))


class TimeDistributedProjection(tf.keras.layers.Layer):

    def __init__(self, num_outputs, name=None):
        """Set up a low-order projection (embedding) by flattening the batch and time dims and matmul

        :param name: The name for this scope
        :param num_outputs: The number of feature maps out
        """
        super(TimeDistributedProjection, self).__init__(True, name)
        self.output_dim = num_outputs
        self.W = None
        self.b = None

    def build(self, input_shape):

        nx = int(input_shape[-1])
        self.W = self.add_variable("W", [nx, self.output_dim])
        self.b = self.add_variable("b", [self.output_dim], initializer=tf.constant_initializer(0.0))
        super(TimeDistributedProjection, self).build(input_shape)

    def call(self, inputs):
        """Low-order projection (embedding) by flattening the batch and time dims and matmul

        :param inputs: The input tensor
        :return: An output tensor having the same dims as the input, except the last which is `output_dim`
        """
        input_shape = get_shape_as_list(inputs)
        collapse = tf.reshape(inputs, [-1, input_shape[-1]])
        c = tf.matmul(collapse, self.W) + self.b
        c = tf.reshape(c, input_shape[:-1] + [self.output_dim])
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    @property
    def requires_length(self):
        return False


def scaled_dot_product_attention(query, key, value, pdrop=0.0, mask=None, training=False):
    w = tf.matmul(query, key, transpose_b=True)

    w *= tf.rsqrt(tf.to_float(tf.shape(query)[2]))

    if mask is not None:
        w = w * mask + -1e9 * (1 - mask)

    weights = tf.nn.softmax(w, name="attention_weights")
    weights = tf.layers.dropout(weights, pdrop, training=training)
    return tf.matmul(weights, value), weights


def dot_product_attention(query, key, value, pdrop=0.0, mask=None, training=False):
    w = tf.matmul(query, key, transpose_b=True)

    if mask is not None:
        w = w * mask + -1e9 * (1 - mask)

    weights = tf.nn.softmax(w, name="attention_weights")
    weights = tf.layers.dropout(weights, pdrop, training=training)
    return tf.matmul(weights, value), weights


def split_heads(x, num_heads):
    shp = get_shape_as_list(x)
    dsz = shp[-1]
    r = tf.reshape(x, shp[:-1] + [num_heads, dsz // num_heads])
    # (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
    return tf.transpose(r, [0, 2, 1, 3])


def combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    shp = get_shape_as_list(x)
    num_heads, head_sz = shp[-2:]
    new_x_shape = shp[:-2]+[num_heads * head_sz]
    new_x = tf.reshape(x, new_x_shape)
    return new_x


class MultiHeadedAttention(tf.keras.Model):
    """
    Multi-headed attention from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    Multi-headed attention provides multiple looks of low-order projections K, Q and V using an attention function
    (specifically `scaled_dot_product_attention` in the paper.  This allows multiple relationships to be illuminated
    via attention on different positional and representational information from each head.

    The number of heads `h` times the low-order projection dim `d_k` is equal to `d_model` (which is asserted upfront).
    This means that each weight matrix can be simply represented as a linear transformation from `d_model` to `d_model`,
    and partitioned into heads after the fact.

    Finally, an output projection is applied which brings the output space back to `d_model`, in preparation for the
    sub-sequent `FFN` sub-layer.

    There are 3 uses of multi-head attention in the Transformer.
    For encoder-decoder layers, the queries come from the previous decoder layer, and the memory keys come from
    the encoder.  For encoder layers, the K, Q and V all come from the output of the previous layer of the encoder.
    And for self-attention in the decoder, K, Q and V all come from the decoder, but here it is masked to prevent using
    future values
    """
    def __init__(self, h, d_model, dropout=0.1, scale=False):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_Q = TimeDistributedProjection(d_model)
        self.w_K = TimeDistributedProjection(d_model)
        self.w_V = TimeDistributedProjection(d_model)
        self.w_O = TimeDistributedProjection(d_model)
        self.attn_fn = scaled_dot_product_attention if scale else dot_product_attention
        self.attn = None
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, qkv, training=False, mask=None):
        query, key, value = qkv

        # (B, H, T, D)
        query = split_heads(self.w_Q(query), self.h)
        key = split_heads(self.w_K(key), self.h)
        value = split_heads(self.w_V(value), self.h)
        x, self.attn = self.attn_fn(query, key, value, mask=mask, pdrop=self.dropout)
        x = combine_heads(x)
        return self.w_O(x)


class TransformerEncoder(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None, name=None):
        super(TransformerEncoder, self).__init__(name=name)
        if d_ff is None:
            d_ff = 4*d_model
        self.ln1 = LayerNorm(name='ln_1')
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.ln2 = LayerNorm(name='ln_2')
        self.feed_forward = FFN(d_model, pdrop, activation_type, d_ff, name='ffn')

    def call(self, inputs, training=False, mask=None):
        x = inputs

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x), training, mask), training)

        x = self.ln2(x)

        x = x + self.dropout(self.feed_forward(x, training, mask), training)
        return x


class TransformerDecoder(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None, name=None):
        super(TransformerEncoder, self).__init__(name=name)
        if d_ff is None:
            d_ff = 4*d_model

        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.ln1 = LayerNorm(name='ln_1')
        self.ln2 = LayerNorm(name='ln_2')
        self.ln3 = LayerNorm(name='ln_3')
        self.feed_forward = FFN(d_model, pdrop, activation_type, d_ff, name='ffn')

    def call(self, inputs, training=False, mask=None):
        memory, x = inputs
        x = self.ln1(x)
        src_mask = None
        tgt_mask = None
        if mask is not None:
            src_mask, tgt_mask = mask

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x), training, tgt_mask))

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory), src_mask))

        x = self.ln3(x)
        x = x + self.dropout(self.feed_forward(x, training, mask))
        return x


class TransformerEncoderStack(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, layers=1, activation_type='relu', d_ff=None, name=None, **kwargs):
        super(TransformerEncoderStack, self).__init__(name=name)
        self.encoders = []
        self.ln = LayerNorm(name='ln_out')
        for i in range(layers):
            self.encoders.append(TransformerEncoder(d_model, num_heads, pdrop, scale, activation_type, d_ff))

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for layer in self.encoders:
            x = layer(x, training, mask)
        return self.ln(x)


class TransformerDecoderStack(tf.keras.Model):
    def __init__(self, d_model, num_heads, pdrop, scale=True, layers=1, activation_type='relu', d_ff=None, name=None, **kwargs):
        super(TransformerDecoderStack, self).__init__()
        self.decoders = []
        self.ln = LayerNorm(name='ln_out')
        for i in range(layers):
            self.decoders.append(TransformerDecoder(d_model, num_heads, pdrop, scale, activation_type, d_ff, name))

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for layer in self.decoders:
            x = layer(x, training, mask)
        return self.ln(x)


class FFN(tf.keras.Model):
    """
    FFN from https://arxiv.org/abs/1706.03762 via http://nlp.seas.harvard.edu/2018/04/03/attention.html

    The `FFN` layer is block in the Transformer that follows multi-headed self-attention.  It consists
    of an expansion from `d_model` to `d_ff` (with sub-sequent relu and dropout), followed by a squeeze
    layer that pushes it back to `d_model`.  In the `tensor2tensor` codebase, this is implemented as convolution of
    size 1 over the temporal sequence, which is equivalent, but in PyTorch, we dont need to do anything explicitly,
    thanks to https://github.com/pytorch/pytorch/pull/1935!

    """
    def __init__(self, d_model, pdrop, activation_type='relu', d_ff=None, name=None):
        """Constructor, takes in model size (which is the external currency of each block) and the feed-forward size

        :param d_model: The model size.  This is the size passed through each block
        :param d_ff: The feed-forward internal size, which is typical 4x larger, used internally
        :param pdrop: The probability of dropping output
        """
        super(FFN, self).__init__(name=name)
        if d_ff is None:
            d_ff = 4 * d_model
        self.expansion = TimeDistributedProjection(d_ff)
        self.squeeze = TimeDistributedProjection(d_model)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.act = tf.keras.layers.Activation(activation_type)

    def call(self, inputs, training=False, mask=None):
        return self.squeeze(self.dropout(self.act(self.expansion(inputs)), training))


class EmbedPoolStackModel(tf.keras.Model):

    def __init__(self, nc, embeddings, pool_model, stack_model=None):
        super(EmbedPoolStackModel, self).__init__()
        assert isinstance(embeddings, dict)

        self.embed_model = EmbeddingsStack(embeddings)

        self.pool_requires_length = False
        if hasattr(pool_model, 'requires_length'):
            self.pool_requires_length = pool_model.requires_length
        self.output_layer = tf.keras.layers.Dense(nc)
        self.pool_model = pool_model
        self.stack_model = stack_model

    def call(self, inputs, training=None, mask=None):
        lengths = inputs.get('lengths')

        embedded = self.embed_model(inputs)

        if self.pool_requires_length:
            embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled) if self.stack_model is not None else pooled
        return self.output_layer(stacked)


class CRF(tf.keras.layers.Layer):

    def __init__(self, num_tags, constraint_mask=None, name=None):
        """Initialize the object.
        :param n_tags: int, The number of tags in your output (emission size)
        :param idxs: Tuple(int. int), The index of the start and stop symbol
            in emissions.
        :param batch_first: bool, if the input [B, T, ...] or [T, B, ...]
        :param mask: torch.ByteTensor, Constraints on the transitions [1, N, N]

        Note:
            if idxs is none then the CRF adds these symbols to the emission
            vectors and n_tags is assumed to be the number of output tags.
            if idxs is not none then the first element is assumed to be the
            start index and the second idx is assumed to be the end index. In
            this case n_tags is assumed to include the start and end symbols.
        """
        super(CRF, self).__init__(name=name)

        self.A = self.add_variable("transitions_raw", shape=(num_tags, num_tags), dtype=tf.float32)
        self.num_tags = num_tags
        self.mask = None
        self.inv_mask = None
        if constraint_mask is not None:
            self.mask, inv_mask = constraint_mask
            self.inv_mask = inv_mask * tf.constant(-1e4)

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return (self.A * self.mask) + self.inv_mask
        return self.A

    def score_sentence(self, unary, tags, lengths):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [T, B, N]
        :param tags: torch.LongTensor: [T, B]
        :param lengths: torch.LongTensor: [B]
        :param batzh_size: int: B
        :param min_length: torch.LongTensor: []

        :return: torch.FloatTensor: [B]
        """
        return tf.contrib.crf.crf_sequence_score(unary, tags, lengths, self.transitions)

    def call(self, inputs, training=False, mask=None):

        unary, lengths = inputs
        if training:
            return tf.contrib.crf.crf_log_norm(unary, lengths, self.transitions)

        else:
            return self.decode(unary, lengths)

    def decode(self, unary, lengths):
        """Do Viterbi decode on a batch.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param lengths: torch.LongTensor: [B]

        :return: List[torch.LongTensor]: [B] the paths
        :return: torch.FloatTensor: [B] the path score
        """
        bsz = tf.shape(unary)[0]
        lsz = self.num_tags
        np_gos = np.full((1, 1, lsz), -1e4, dtype=np.float32)
        np_gos[:, :, Offsets.GO] = 0
        gos = tf.constant(np_gos)
        start = tf.tile(gos, [bsz, 1, 1])
        probv = tf.concat([start, unary], axis=1)
        viterbi, _ = tf.contrib.crf.crf_decode(probv, self.transitions, lengths + 1)
        return tf.identity(viterbi[:, 1:], name="best")

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: torch.FloatTensor: [T, B, N] or [B, T, N]
        :param tags: torch.LongTensor: [T, B] or [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        fwd_score = self((unary, lengths), training=True)
        gold_score = self.score_sentence(unary, tags, lengths)
        log_likelihood = gold_score - fwd_score
        return -tf.reduce_mean(log_likelihood)
