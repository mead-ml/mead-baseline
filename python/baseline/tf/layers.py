import tensorflow as tf
from baseline.utils import listify
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


class ParallelConv(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(self, dsz, motsz, filtsz, activation='relu'):

        super(ParallelConv, self).__init__()
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


def stacked_lstm(hsz, pdrop, nlayers, variational=False, training=False):
    """Produce a stack of LSTMs with dropout performed on all but the last layer.

    :param hsz: (``int``) The number of hidden units per LSTM
    :param pdrop: (``int``) The probability of dropping a unit value during dropout
    :param nlayers: (``int``) The number of layers of LSTMs to stack
    :param variational (``bool``) variational recurrence is on
    :param training (``bool``) Are we training? (defaults to ``False``)
    :return: a stacked cell
    """
    if variational:
        return tf.contrib.rnn.MultiRNNCell(
            [lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=training) for _ in range(nlayers)],
            state_is_tuple=True
        )
    return tf.contrib.rnn.MultiRNNCell(
        [lstm_cell_w_dropout(hsz, pdrop, training=training) if i < nlayers - 1 else lstm_cell(hsz) for i in range(nlayers)],
        state_is_tuple=True
    )


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


def rnn_cell_w_dropout(hsz, pdrop, rnntype, st=None, variational=False, training=False):

    """Produce a single RNN cell with dropout

    :param hsz: (``int``) The number of hidden units per LSTM
    :param rnntype: (``str``): `lstm` or `gru`
    :param pdrop: (``int``) The probability of dropping a unit value during dropout
    :param st: (``bool``) state is tuple? defaults to `None`
    :param variational: (``bool``) Variational recurrence is on
    :param training: (``bool``) Are we training?  Defaults to ``False``
    :return: a cell
    """
    output_keep_prob = tf.contrib.framework.smart_cond(training, lambda: 1.0 - pdrop, lambda: 1.0)
    state_keep_prob = tf.contrib.framework.smart_cond(training, lambda: 1.0 - pdrop if variational else 1.0, lambda: 1.0)
    cell = rnn_cell(hsz, rnntype, st)
    output = tf.contrib.rnn.DropoutWrapper(cell,
                                           output_keep_prob=output_keep_prob,
                                           state_keep_prob=state_keep_prob,
                                           variational_recurrent=variational,
                                           dtype=tf.float32)
    return output


def multi_rnn_cell_w_dropout(hsz, pdrop, rnntype, num_layers, variational=False, training=False):
    """Produce a stack of RNNs with dropout performed on all but the last layer.

    :param hsz: (``int``) The number of hidden units per RNN
    :param pdrop: (``int``) The probability of dropping a unit value during dropout
    :param rnntype: (``str``) The type of RNN to use - `lstm` or `gru`
    :param num_layers: (``int``) The number of layers of RNNs to stack
    :param training: (``bool``) Are we training? Defaults to ``False``
    :return: a stacked cell
    """
    if variational:
        return tf.contrib.rnn.MultiRNNCell(
            [rnn_cell_w_dropout(hsz, pdrop, rnntype, variational=variational, training=training) for _ in range(num_layers)],
            state_is_tuple=True
        )
    return tf.contrib.rnn.MultiRNNCell(
        [rnn_cell_w_dropout(hsz, pdrop, rnntype, training=training) if i < num_layers - 1 else rnn_cell_w_dropout(hsz, 1.0, rnntype) for i in range(num_layers)],
        state_is_tuple=True
    )


class LSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, pdrop, nlayers, variational=False, output_fn=None, requires_length=True):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param variational (``bool``) variational recurrence is on
        :param training (``bool``) Are we training? (defaults to ``False``)
        :return: a stacked cell
        """
        super(LSTMEncoder, self).__init__()
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

    def call(self, inputs, **kwargs):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length

class BiLSTMEncoder(tf.keras.Model):

    def __init__(self, hsz, pdrop, nlayers, variational=False, output_fn=None, requires_length=True):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param hsz: (``int``) The number of hidden units per LSTM
        :param pdrop: (``int``) The probability of dropping a unit value during dropout
        :param nlayers: (``int``) The number of layers of LSTMs to stack
        :param variational (``bool``) variational recurrence is on
        :param training (``bool``) Are we training? (defaults to ``False``)
        :return: a stacked cell
        """
        super(BiLSTMEncoder, self).__init__()
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

    def call(self, inputs, **kwargs):
        inputs, lengths = tensor_and_lengths(inputs)
        rnnout, hidden = tf.nn.bidirectional_dynamic_rnn(self.rnn_fwd, self.rnn_bwd, input, sequence_length=lengths, dtype=tf.float32)
        return self.output_fn(rnnout, hidden)

    @property
    def requires_length(self):
        return self._requires_length


class EmbeddingsStack(tf.keras.Model):

    def __init__(self, embeddings_dict, requires_length=False):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """

        super(EmbeddingsStack, self).__init__()
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

    def __init__(self, hsz, activation='relu', pdrop_value=0.5, init=None):
        super(DenseStack, self).__init__()
        hszs = listify(hsz)
        self.layer_stack = [tf.keras.layers.Dense(hsz, kernel_initializer=init, activation=activation) for hsz in hszs]
        self.dropout = tf.keras.layers.Dropout(pdrop_value)

    def call(self, inputs, training=False):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param pooled: The fixed representation of the model
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

    def __init__(self, input_size):
        super(Highway, self).__init__()
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

    def __init__(self, layer=None):
        super(ResidualBlock, self).__init__()
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

        :param x: The input tensor
        :param name: The name for this scope
        :param filters: The number of feature maps out
        :param w_init: An optional weight initializer
        :param b_init: An optional bias initializer
        :return:
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


#class ClassificationSequentialModel(tf.keras.Model):
#
#    def __init__(self, nc, sequence):
#        super(ClassificationSequentialModel, self).__init__()
#        self.sequence = sequence
#        self.requires_length = [isinstance(module, (LSTMEncoder, BiLSTMEncoder)) for module in sequence]
#
#    def call(self, inputs, training=None, mask=None):
#        lengths = inputs.pop('lengths')
#        for i, module in enumerate(self.sequence):
#            if self.requires_length[i]:
#                x = module((x, lengths))
#            else:
#                x = module(x)
#        x = self.output_layer(x)
#        return x


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



