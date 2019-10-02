import tensorflow as tf
import numpy as np
from baseline.utils import listify, Offsets, wraps
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


# Mapped
def tensor_and_lengths(inputs):
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None  ##tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), axis=1)

    return in_tensor, lengths


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


# Mapped
def get_activation(name='relu'):
    if name is None or name == 'ident':
        return tf.nn.identity
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
        return tf.identity
    if name == 'leaky_relu':
        return tf.nn.leaky_relu
    return tf.nn.relu


# Mapped
class ConvEncoder(tf.keras.Model):
    def __init__(self, outsz, filtsz, pdrop, activation='relu'):
        super(ConvEncoder, self).__init__()
        self.output_dim = outsz
        self.conv = tf.keras.layers.Conv1D(filters=outsz, kernel_size=filtsz, padding='same')
        self.act = get_activation(activation)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        conv_out = self.act(self.conv(inputs))
        return self.dropout(conv_out, TRAIN_FLAG())


# Mapped
class ConvEncoderStack(tf.keras.Model):

    def __init__(self, outsz, filtsz, pdrop, layers=1, activation='relu'):
        super(ConvEncoderStack, self).__init__()

        first_layer = ConvEncoder(outsz, filtsz, pdrop, activation)
        self.layers.append(first_layer)
        for i in range(layers-1):
            subsequent_layer = ResidualBlock(ConvEncoder(outsz, filtsz, pdrop, activation))
            self.layers.append(subsequent_layer)

    def call(self, inputs):
        for layer in self.layers:
            x = layer(x)
        return x


# Mapped
class ParallelConv(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(self, insz, outsz, filtsz, activation='relu', name=None, **kwargs):
        """Do parallel convolutions with multiple filter widths and max-over-time pooling.

        :param filtsz: The list of filter widths to use.
        :param dsz: The depths of the input (H).
        :param motsz: The number of conv filters to use (can be an int or a list to allow for various sized filters)
        :param activation: (``str``) The name of the activation function to use (`default='relu`)
        """
        super(ParallelConv, self).__init__(name=name)
        self.Ws = []
        self.bs = []
        self.activation = get_activation(activation)

        motsz = outsz
        if not isinstance(outsz, list):
            motsz = [outsz] * len(filtsz)

        for fsz, cmotsz in zip(filtsz, motsz):
            kernel_shape = [1, int(fsz), int(insz), int(cmotsz)]
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

# Mapped
def rnn_ident(output, hidden):
    return output, hidden

# Mapped
def rnn_signal(output, hidden):
    return output

# Mapped
def rnn_hidden(output, output_state):
    output_state = output_state[-1].h
    return output_state

# Mapped
def rnn_bi_hidden(output, output_state):
    fw_final_state, bw_final_state = output_state
    output_state = fw_final_state[-1].h + bw_final_state[-1].h
    return output_state


def lstm_cell(hsz, forget_bias=1.0, **kwargs):
    """Produce a single cell with no dropout
    :param hsz: (``int``) The number of hidden units per LSTM
    :param forget_bias: (``int``) Defaults to 1
    :return: a cell
    """
    num_proj = kwargs.get('projsz')
    if num_proj and num_proj == hsz:
        num_proj = None
    cell = tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True, num_proj=num_proj)
    skip_conn = bool(kwargs.get('skip_conn', False))
    return tf.nn.rnn_cell.ResidualWrapper(cell) if skip_conn else cell


def lstm_cell_w_dropout(hsz, pdrop, forget_bias=1.0, variational=False, training=False, **kwargs):
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
    num_proj = kwargs.get('projsz')
    cell = tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True, num_proj=num_proj)
    skip_conn = bool(kwargs.get('skip_conn', False))
    cell = tf.nn.rnn_cell.ResidualWrapper(cell) if skip_conn else cell
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


# Mapped
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, axis=-1, name=None, **kwargs):
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


# Mapped
class LSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, nlayers, pdrop=0.0, variational=False, output_fn=None, requires_length=True, name=None,
                 dropout_in_single_layer=False, skip_conn=False, projsz=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param variational: (``bool``) variational recurrence is on
        :param output_fn: A function that filters output to decide what to return
        :param requires_length: (``bool``) Does the input require an input length (defaults to ``True``)
        :param name: (``str``) Optional, defaults to `None`
        :return: a stacked cell
        """
        super(LSTMEncoder, self).__init__(name=name)
        self._requires_length = requires_length

        if variational or dropout_in_single_layer:
            self.rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pdrop,
                                                                        variational=variational,
                                                                        training=TRAIN_FLAG(),
                                                                        skip_conn=skip_conn,
                                                                        projsz=projsz) for _ in
                                                    range(nlayers)],
                                                   state_is_tuple=True
                                                   )
        else:
            self.rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(),
                                     skip_conn=skip_conn, projsz=projsz) if i < nlayers - 1 else lstm_cell(hsz, skip_conn=skip_conn, projsz=projsz) for i in range(nlayers)],
                state_is_tuple=True
            )
        self.output_fn = rnn_ident if output_fn is None else output_fn

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length


class LSTMEncoderWithState(LSTMEncoder):
    def __init__(self, hsz, nlayers, pdrop=0.0, variational=False, output_fn=None, name=None, dropout_in_single_layer=True, **kwargs):
        super(LSTMEncoderWithState, self).__init__(hsz, nlayers, pdrop, variational, output_fn, False, name, dropout_in_single_layer, **kwargs)

        #h = tf.reshape(tf.concat(rnnout, 1), [-1, self.hsz])
        #self.final_state = state

    def zero_state(self, batchsz):
        return self.rnn.zero_state(batchsz, tf.float32)

    def call(self, inputs):

        inputs, hidden = inputs
        rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, initial_state=hidden, dtype=tf.float32)
        return self.output_fn(rnnout, hidden)


# Mapped
class BiLSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, nlayers, pdrop=0.0, variational=False, output_fn=None, requires_length=True, name=None,  skip_conn=False, projsz=None, **kwargs):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param variational (``bool``) variational recurrence is on
        :param training (``bool``) Are we training? (defaults to ``False``)
        :return: a stacked cell
        """
        super(BiLSTMEncoder, self).__init__(name=name)
        self._requires_length = requires_length
        if variational:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz) for _ in
                                                        range(nlayers)],
                                                       state_is_tuple=True
                                                       )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz) for _ in
                 range(nlayers)],
                state_is_tuple=True
            )
        else:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz) if i < nlayers - 1 else lstm_cell(hsz, skip_conn=skip_conn, projsz=projsz) for i in range(nlayers)],
                state_is_tuple=True
            )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz) if i < nlayers - 1 else lstm_cell(hsz) for i in
                 range(nlayers)],
                state_is_tuple=True
            )
        self.output_fn = rnn_ident if output_fn is None else output_fn
        print(self.output_fn)

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.bidirectional_dynamic_rnn(self.fwd_rnn, self.bwd_rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        rnnout = tf.concat(axis=2, values=rnnout)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length


class EmbeddingsStack(tf.keras.Model):

    def __init__(self, embeddings_dict, dropout_rate=0.0, requires_length=False, name=None, **kwargs):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """

        super(EmbeddingsStack, self).__init__(name=name)
        self.embeddings = embeddings_dict
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
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
        return self.dropout(word_embeddings, TRAIN_FLAG())

    @property
    def dsz(self):
        total_dsz = 0
        for embeddings in self.embeddings.values():
            total_dsz += embeddings.get_dsz()
        return total_dsz

    @property
    def requires_length(self):
        return self.requires_length

    @property
    def output_dim(self):
        return self.dsz


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

    def call(self, inputs):
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
            x = self.dropout(x, TRAIN_FLAG())
        return x

    @property
    def requires_length(self):
        return False


class WithDropout(tf.keras.Model):

    def __init__(self, layer, pdrop=0.5):
        super(WithDropout, self).__init__()
        self.layer = layer
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        return self.dropout(self.layer(inputs), TRAIN_FLAG())

    @property
    def output_dim(self):
        return self.layer.output_dim


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


def scaled_dot_product_attention(query, key, value, pdrop=0.0, mask=None):
    w = tf.matmul(query, key, transpose_b=True)

    w *= tf.rsqrt(tf.to_float(tf.shape(query)[2]))

    if mask is not None:
        w = w * mask + -1e9 * (1 - mask)

    weights = tf.nn.softmax(w, name="attention_weights")
    weights = tf.layers.dropout(weights, pdrop, training=TRAIN_FLAG())
    return tf.matmul(weights, value), weights


def dot_product_attention(query, key, value, pdrop=0.0, mask=None):
    w = tf.matmul(query, key, transpose_b=True)

    if mask is not None:
        w = w * mask + -1e9 * (1 - mask)

    weights = tf.nn.softmax(w, name="attention_weights")
    weights = tf.layers.dropout(weights, pdrop, training=TRAIN_FLAG())
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
        self.dropout = dropout

    def call(self, qkv, mask=None):
        query, key, value = qkv

        # (B, H, T, D)
        query = split_heads(self.w_Q(query), self.h)
        key = split_heads(self.w_K(key), self.h)
        value = split_heads(self.w_V(value), self.h)
        x, self.attn = self.attn_fn(query, key, value, mask=mask, pdrop=self.dropout)
        x = combine_heads(x)
        return self.w_O(x)


class TransformerEncoder(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, activation='relu', d_ff=None, name=None):
        super(TransformerEncoder, self).__init__(name=name)
        if d_ff is None:
            d_ff = 4 * d_model
        self.ln1 = LayerNorm(name='ln_1')
        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.ln2 = LayerNorm(name='ln_2')
        self.feed_forward = FFN(d_model, pdrop, activation, d_ff, name='ffn')

    def call(self, inputs, mask=None):
        x = inputs

        x = self.ln1(x, mask=mask)
        x = x + self.dropout(self.self_attn((x, x, x), mask=mask), TRAIN_FLAG())

        x = self.ln2(x, mask=mask)

        x = x + self.dropout(self.feed_forward(x), TRAIN_FLAG())
        return x


class TransformerDecoder(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, activation='relu', d_ff=None, name=None):
        super(TransformerEncoder, self).__init__(name=name)
        if d_ff is None:
            d_ff = 4 * d_model

        self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.ln1 = LayerNorm(name='ln_1')
        self.ln2 = LayerNorm(name='ln_2')
        self.ln3 = LayerNorm(name='ln_3')
        self.feed_forward = FFN(d_model, pdrop, activation, d_ff, name='ffn')

    def call(self, inputs, mask=None):
        memory, x = inputs
        x = self.ln1(x)
        src_mask = None
        tgt_mask = None
        if mask is not None:
            src_mask, tgt_mask = mask

        x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x), tgt_mask), TRAIN_FLAG())

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory), src_mask), TRAIN_FLAG())

        x = self.ln3(x)
        x = x + self.dropout(self.feed_forward(x), TRAIN_FLAG())
        return x


class TransformerEncoderStack(tf.keras.Model):

    def __init__(self, d_model, num_heads, pdrop, scale=True, layers=1, activation='relu', d_ff=None, name=None, **kwargs):
        super(TransformerEncoderStack, self).__init__(name=name)
        self.encoders = []
        self.ln = LayerNorm(name='ln_out')
        for i in range(layers):
            self.encoders.append(TransformerEncoder(d_model, num_heads, pdrop, scale, activation, d_ff))

    def call(self, inputs, mask=None):
        x = inputs
        for layer in self.encoders:
            x = layer(x, mask=mask)
        return self.ln(x)


class TransformerDecoderStack(tf.keras.Model):
    def __init__(self, d_model, num_heads, pdrop, scale=True, layers=1, activation='relu', d_ff=None, name=None, **kwargs):
        super(TransformerDecoderStack, self).__init__()
        self.decoders = []
        self.ln = LayerNorm(name='ln_out')
        for i in range(layers):
            self.decoders.append(TransformerDecoder(d_model, num_heads, pdrop, scale, activation, d_ff, name))

    def call(self, inputs, mask=None):
        x = inputs
        for layer in self.decoders:
            x = layer(x, mask=mask)
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
    def __init__(self, d_model, pdrop, activation='relu', d_ff=None, name=None):
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
        self.act = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        return self.squeeze(self.dropout(self.act(self.expansion(inputs)), TRAIN_FLAG()))


class TaggerGreedyDecoder(tf.keras.layers.Layer):

    def __init__(self, num_tags, constraint_mask=None, name=None):
        super(TaggerGreedyDecoder, self).__init__(name=name)
        self.num_tags = num_tags
        self.inv_mask = None
        if constraint_mask is not None:
            _, inv_mask = constraint_mask
            self.inv_mask = inv_mask * tf.constant(-1e4)

        self.A = self.add_variable("transitions_raw", shape=(num_tags, num_tags), dtype=tf.float32, init='zeros', trainable=False)

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return tf.nn.log_softmax(self.A + self.inv_mask)
        return self.A

    def neg_log_loss(self, unary, tags, lengths):
        # Cross entropy loss
        mask = tf.sequence_mask(lengths)
        cross_entropy = tf.one_hot(tags, self.num_tags, axis=-1) * tf.log(tf.nn.softmax(unary))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def call(self, inputs, training=False, mask=None):

        unary, lengths = inputs

        if self.inv_mask is not None:
            bsz = tf.shape(unary)[0]
            lsz = self.num_tags
            np_gos = np.full((1, 1, lsz), -1e4, dtype=np.float32)
            np_gos[:, :, Offsets.GO] = 0
            gos = tf.constant(np_gos)
            start = tf.tile(gos, [bsz, 1, 1])
            probv = tf.concat([start, unary], axis=1)
            viterbi, _ = tf.contrib.crf.crf_decode(probv, self.transitions, lengths + 1)
            return tf.identity(viterbi[:, 1:], name="best")
        else:
            return tf.argmax(self.probs, 2, name="best")


class CRF(tf.keras.layers.Layer):

    def __init__(self, num_tags, constraint_mask=None, name=None):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint_mask: torch.ByteTensor, Constraints on the transitions [1, N, N]
        :param name: str, Optional name, defaults to `None`
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

    def call(self, inputs, training=False):

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
        start = tf.nn.log_softmax(start, axis=-1)

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


class TagSequenceModel(tf.keras.Model):

    def __init__(self, nc, embeddings, transducer, decoder=None, name=None):
        super(TagSequenceModel, self).__init__(name=name)
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings
        self.transducer_model = transducer
        self.proj_layer = TimeDistributedProjection(nc)
        decoder_model = CRF(nc) if decoder is None else decoder
        self.decoder_model = decoder_model

    def transduce(self, inputs):
        lengths = inputs.get('lengths')

        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced, lengths):
        return self.decoder_model((transduced, lengths))

    def call(self, inputs, training=None):
        transduced = self.transduce(inputs)
        return self.decode(transduced, inputs.get('lengths'))

    def neg_log_loss(self, unary, tags, lengths):
        return self.decoder_model.neg_log_loss(unary, tags, lengths)


class LangSequenceModel(tf.keras.Model):

    def __init__(self, nc, embeddings, transducer, decoder=None, name=None):
        super(LangSequenceModel, self).__init__(name=name)
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings
        self.transducer_model = transducer

        self.output_layer = TimeDistributedProjection(nc)
        self.decoder_model = decoder

    def call(self, inputs):

        h = inputs.get('h')

        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        ##transduced, hidden = self.transducer_model(embedded, training)
        transduced = self.output_layer(transduced)
        return transduced, hidden


class EmbedPoolStackModel(tf.keras.Model):

    def __init__(self, nc, embeddings, pool_model, stack_model=None):
        super(EmbedPoolStackModel, self).__init__()
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings

        self.pool_requires_length = False
        if hasattr(pool_model, 'requires_length'):
            self.pool_requires_length = pool_model.requires_length
        self.output_layer = tf.keras.layers.Dense(nc)
        self.pool_model = pool_model
        self.stack_model = stack_model

    def call(self, inputs):
        lengths = inputs.get('lengths')

        embedded = self.embed_model(inputs)

        if self.pool_requires_length:
            embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled) if self.stack_model is not None else pooled
        return self.output_layer(stacked)

    def get_config(self):
        #base_config = super(EmbedPoolStackModel, self).get_config()
        return {} #base_config


class FineTuneModel(tf.keras.Model):

    def __init__(self, nc, embeddings, stack_model=None):
        super(FineTuneModel, self).__init__()
        if isinstance(embeddings, dict):
            self.finetuned = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.finetuned = embeddings
        self.stack_model = stack_model
        self.output_layer = tf.keras.layers.Dense(nc)

    def call(self, inputs, training=None, mask=None):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)

    def get_config(self):
        #base_config = super(FineTuneModel, self).get_config()
        return {} # base_config


def highway_conns(inputs, wsz_all, n):
    """Produce one or more highway connection layers

    :param inputs: The sub-graph input
    :param wsz_all: The number of units
    :param n: How many layers of gating
    :return: graph output
    """
    x = inputs
    for i in range(n):
        x = Highway(wsz_all)(x)
    return x


def skip_conns(inputs, wsz_all, n, activation_fn='relu'):
    x = inputs
    for i in range(n):
        x = SkipConnection(wsz_all, activation_fn)(x)
    return x


def layer_norm(input, name, axis=[-1]):
    return LayerNorm(name=name, axis=axis)(input)


def parallel_conv(input_, filtsz, dsz, motsz, activation_fn='relu'):
    return ParallelConv(dsz, motsz, filtsz, activation_fn)(input_)


def time_distributed_projection(x, name, filters):
    return TimeDistributedProjection(filters, name)(x)


def char_word_conv_embeddings(char_vec, filtsz, char_dsz, nfeats, activation_fn=tf.nn.tanh, gating=skip_conns, num_gates=1):
    """This wrapper takes in a character vector as input and performs parallel convolutions on it, followed by a
    pooling operation and optional residual or highway connections

    :param char_vec: The vector input
    :param filtsz: A list or scalar containing filter sizes for each parallel filter
    :param char_dsz: The character dimension size
    :param nfeats: A list or scalar of the number of pooling units for each filter operation
    :param activation_fn: A function for activation (`tf.nn.tanh` etc)
    :param gating: A gating function to apply to the output
    :param num_gates: The number of gates to apply
    :return: The embedding output, the full number of units
    """
    if isinstance(nfeats, (list, tuple)):
        wsz_all = np.sum(nfeats)
    else:
        wsz_all = len(filtsz) * nfeats
    combine = parallel_conv(char_vec, filtsz, char_dsz, nfeats, activation_fn)
    joined = gating(combine, wsz_all, num_gates)
    return joined, wsz_all

def create_session():
    """This function protects against TF allocating all the memory

    Some combination of cuDNN 7.6 with CUDA 10 on TF 1.13 with RTX cards
    allocate additional memory which isnt available since TF by default
    hogs it all.


    This also provides an abstraction that can be extended later to offer
    more config params that raw `tf.Session()` calls dont

    :return: A `tf.Session`
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def reload_lower_layers(sess, checkpoint):
    """
    Get the intersection of all non-output layers and declared vars in this graph and restore them

    :param sess: (`tf.Session`) A tensorflow session to restore from
    :param checkpoint: (`str`) checkpoint to read from
    :return: None
    """
    latest = tf.train.latest_checkpoint(checkpoint)
    print('Reloading ' + latest)
    model_vars = set([t[0] for t in tf.train.list_variables(latest)])
    g = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
    g = [v for v in g if not v.op.name.startswith('OptimizeLoss')]
    g = [v for v in g if not v.op.name.startswith('output/')]
    g = [v for v in g if v.op.name in model_vars]
    saver = tf.train.Saver(g)
    saver.restore(sess, latest)


def tf_device_wrapper(func):
    @wraps(func)
    def with_device(*args, **kwargs):
        device = kwargs.get('device', 'default')
        if device == 'cpu' and 'sess' not in kwargs:
            g = tf.Graph()
            sess = tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True, device_count={'CPU': 1, 'GPU': 0}))
            kwargs['sess'] = sess
            return func(*args, **kwargs)
        return func(*args, **kwargs)
    return with_device
