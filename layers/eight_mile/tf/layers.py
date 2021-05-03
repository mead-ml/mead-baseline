import logging
import tensorflow as tf
from tensorflow.python.lib.io.file_io import FileIO
import numpy as np
import os
from eight_mile.utils import listify, Offsets, wraps, get_version, is_sequence, transition_mask as transition_mask_np, get_num_gpus_multiworker
from typing import Optional, Union, List, Dict, Any, Tuple
import contextlib
import math
import json

BASELINE_TF_TRAIN_FLAG = None
LOGGER = logging.getLogger('mead.layers')


@contextlib.contextmanager
def autograph_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


def set_tf_eager_mode(prefer_eager: bool = False):
    tf_version = get_version(tf)
    if prefer_eager and tf_version < 2:
        LOGGER.info('user requesting eager on 1.x')
        tf.compat.v1.enable_eager_execution()
    elif not prefer_eager and tf_version >= 2:
        LOGGER.info('User requesting eager disabled on 2.x')
        tf.compat.v1.disable_eager_execution()


def set_tf_eager_debug(debug: bool = False):
    if tf.executing_eagerly():
        if debug:
            tf.config.experimental_run_functions_eagerly(debug)


def patch_dist_strategy(strategy):
    TF_VERSION = get_version(tf)
    if TF_VERSION < 2.2:
        strategy.run = strategy.experimental_run_v2


def read_yaml_tf(filepath: str) -> Dict:
    """Read a YAML file using TensorFlow API (supports GCP remote files)
    :param: filepath, The file name to load
    :return: dict, The read yaml object
    """
    file_stream = FileIO(filepath, mode='r')
    as_str = file_stream.read()

    import yaml
    from distutils.version import LooseVersion

    if LooseVersion(yaml.__version__) >= LooseVersion("5.1"):
        return yaml.load(as_str, Loader=yaml.FullLoader)
    return yaml.load(as_str)


def read_json_tf(filepath: str) -> Dict:
    """Read a JSON file using TensorFlow API (supports GCP remote files)
    :param: filepath, The file name to load
    :return: dict, The read yaml object
    """
    file_stream = FileIO(filepath, mode='r')
    as_str = file_stream.read()
    return json.loads(as_str)


def set_tf_log_level(ll):
    # 0     | DEBUG            | [Default] Print all messages
    # 1     | INFO             | Filter out INFO messages
    # 2     | WARNING          | Filter out INFO & WARNING messages
    # 3     | ERROR            | Filter out all messages
    import os

    TF_VERSION = get_version(tf)
    if TF_VERSION < 2:
        import tensorflow.compat.v1.logging as tf_logging
    else:
        from absl import logging as tf_logging
    tf_ll = tf_logging.WARN
    tf_cpp_ll = 1
    ll = ll.lower()
    if ll == "debug":
        tf_ll = tf_logging.DEBUG
        tf_cpp_ll = 0
    if ll == "info":
        tf_cpp_ll = 0
        tf_ll = tf_logging.INFO
    if ll == "error":
        tf_ll = tf_logging.ERROR
        tf_cpp_ll = 2
    tf_logging.set_verbosity(tf_ll)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = f"{tf_cpp_ll}"


def SET_TRAIN_FLAG(X):
    global BASELINE_TF_TRAIN_FLAG
    BASELINE_TF_TRAIN_FLAG = X


def TRAIN_FLAG():
    """Create a global training flag on first use"""
    global BASELINE_TF_TRAIN_FLAG
    if BASELINE_TF_TRAIN_FLAG is not None:
        return BASELINE_TF_TRAIN_FLAG

    BASELINE_TF_TRAIN_FLAG = tf.compat.v1.placeholder_with_default(False, shape=(), name="TRAIN_FLAG")
    return BASELINE_TF_TRAIN_FLAG


def create_distribute_strategy(strategy_name, endpoint=None, num_gpus=-1):
    if strategy_name == 'tpu':
        if not endpoint:
            endpoint = os.environ.get('MEAD_TPU')
        elif endpoint == 'colab':
            endpoint = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=endpoint)
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        for tpu in tf.config.list_logical_devices('TPU'):
            LOGGER.info('Device [%s]', tpu.name)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        if strategy_name == "nccl":
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
        elif strategy_name == 'mirror':
            num_gpus = get_num_gpus_multiworker() if num_gpus < 1 else num_gpus
            devices = ['/device:GPU:{}'.format(i) for i in range(num_gpus)]
            strategy = tf.distribute.MirroredStrategy(devices)
        else:
            raise Exception(f"Unsupported strategy {strategy_name}")

        for gpu in tf.config.list_logical_devices('GPU'):
            LOGGER.info('Device [%s]', gpu.name)

    patch_dist_strategy(strategy)
    return strategy


def infer_lengths(tensor, axis=1):
    """Infer the lengths of an input based on the idea the Offsets.PAD was used as the padding token.

    :param tensor: The data to infer the length of, should be either [B, T] or [T, B]
    :param axis: The dimension which contains the sequential signal

    :returns: A Tensor of shape `[B]` that has the lengths for example item in the batch
    """
    if len(tensor.get_shape()) != 2:
        raise ValueError(f"infer_lengths only works with tensors with two dims right now, got {len(tensor.get_shape())}")
    offsets = tf.expand_dims(tf.range(1, tf.shape(tensor)[axis] + 1, dtype=tensor.dtype), 1 - axis)
    non_pad_loc = tf.cast(tf.not_equal(tensor, Offsets.PAD), tensor.dtype)
    return tf.argmax(non_pad_loc * offsets, axis=axis) + 1


# Mapped
def tensor_and_lengths(inputs):
    if isinstance(inputs, (list, tuple)):
        in_tensor, lengths = inputs
    else:
        in_tensor = inputs
        lengths = None  ##tf.reduce_sum(tf.cast(tf.not_equal(inputs, 0), tf.int32), axis=1)

    return in_tensor, lengths


# Get rid of this?
def new_placeholder_dict(train):
    global BASELINE_TF_TRAIN_FLAG

    if train:
        return {BASELINE_TF_TRAIN_FLAG: 1}
    return {}


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))


def swish(x):
    return x * tf.nn.sigmoid(x)


def masked_fill(t, mask, value):
    return t * (1 - tf.cast(mask, t.dtype)) + value * tf.cast(mask, t.dtype)


# https://stackoverflow.com/questions/41897212/how-to-sort-a-multi-dimensional-tensor-using-the-returned-indices-of-tf-nn-top-k
def gather_k(a, b, best_idx, k):
    shape_a = get_shape_as_list(a)
    auxiliary_indices = tf.meshgrid(
        *[tf.range(d) for d in (tf.unstack(shape_a[: (a.get_shape().ndims - 1)]) + [k])], indexing="ij"
    )

    sorted_b = tf.gather_nd(b, tf.stack(auxiliary_indices[:-1] + [best_idx], axis=-1))
    return sorted_b


def get_shape_as_list(x):
    """
    This function makes sure we get a number whenever possible, and otherwise, gives us back
    a graph operation, but in both cases, presents as a list.  This makes it suitable for a
    bunch of different operations within TF, and hides away some details that we really dont care about, but are
    a PITA to get right...

    Borrowed from Alec Radford:
    https://github.com/openai/finetune-transformer-lm/blob/master/utils.py#L38
    """
    try:
        ps = x.get_shape().as_list()
    except:
        ps = x.shape
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def bth2bht(t):
    return tf.transpose(t, [0, 2, 1])


def ident(t):
    return t


def tbh2bht(t):
    return tf.tranpose(t, [0, 2, 1])


def tbh2bth(t):
    return tf.transpose(t, [1, 0, 2])


def bth2tbh(t):
    return t.transpose(t, [1, 0, 2])


# Mapped
def get_activation(name: str = "relu"):
    if name is None or name == "ident":
        return tf.nn.identity
    if name == "softmax":
        return tf.nn.softmax
    if name == "tanh":
        return tf.nn.tanh
    if name == "sigmoid":
        return tf.nn.sigmoid
    if name == "gelu":
        return gelu
    if name == "swish":
        return swish
        return tf.identity
    if name == "leaky_relu":
        return tf.nn.leaky_relu
    return tf.nn.relu


class WithoutLength(tf.keras.layers.Layer):
    """Wrapper layer to remove lengths from the input
    """

    def __init__(self, layer: tf.keras.layers.Layer, name=None):
        super().__init__(name=name)
        self.layer = layer
        self.output_dim = self.layer.output_dim if hasattr(self.layer, "output_dim") else 0

    def call(self, inputs):
        output = self.layer(inputs[0])
        return output


class ConvEncoder(tf.keras.layers.Layer):

    def __init__(self, insz: Optional[int], outsz: int, filtsz: int, pdrop: float = 0.0, activation: str = "relu", bias: bool = True, groups: int = 1, stride: int = 1, name=None):
        super().__init__(name=name)
        self.output_dim = outsz
        self.conv = tf.keras.layers.Conv1D(filters=outsz, kernel_size=filtsz, padding="same", use_bias=bias, groups=groups, strides=stride)

        self.act = get_activation(activation)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        conv_out = self.act(self.conv(inputs))
        return self.dropout(conv_out, TRAIN_FLAG())


class ConvEncoderStack(tf.keras.layers.Layer):
    def __init__(
        self, insz: Optional[int], outsz: int, filtsz: int, nlayers: int = 1, pdrop: float = 0.0, activation: str = "relu",  bias: bool = True, groups: int = 1, name=None
    ):
        super().__init__(name=name)
        self.layers = []
        first_layer = ConvEncoder(insz, outsz, filtsz, pdrop, activation, bias, groups)
        self.layers.append(first_layer)
        for i in range(nlayers - 1):
            subsequent_layer = ResidualBlock(ConvEncoder(insz, outsz, filtsz, pdrop, activation, bias, groups))
            self.layers.append(subsequent_layer)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PassThru(tf.keras.layers.Layer):

    def __init__(self, input_dim):
        super().__init__()
        self.output_dim = input_dim

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs


# Mapped
class ParallelConv(tf.keras.layers.Layer):
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3

    def __init__(
        self,
        insz: Optional[int],
        outsz: Union[int, List[int]],
        filtsz: List[int],
        activation: str = "relu",
        name: Optional[str] = None,
        **kwargs,
    ):
        """Do parallel convolutions with multiple filter widths and max-over-time pooling.

        :param insz: The input size (not required, can pass `None`)
        :param outsz: The output size(s).  Normally this is an int, but it can be a stack of them
        :param filtsz: The list of filter widths to use.
        :param activation: (``str``) The name of the activation function to use (`default='relu`)
        :param name: An optional name
        """
        super().__init__(name=name)
        self.Ws = []
        self.bs = []
        self.activation = get_activation(activation)
        self.insz = insz
        self.filtsz = filtsz
        motsz = outsz
        if not is_sequence(outsz):
            motsz = [outsz] * len(filtsz)
        self.motsz = motsz
        self.output_dim = sum(motsz)


    def build(self, input_shape):
        insz = self.insz if self.insz is not None else tf.shape(input_shape)[-1]
        for fsz, cmotsz in zip(self.filtsz, self.motsz):
            kernel_shape = [1, int(fsz), int(insz), int(cmotsz)]
            self.Ws.append(self.add_weight(f"cmot-{fsz}/W", shape=kernel_shape))
            self.bs.append(
                self.add_weight(f"cmot-{fsz}/b", shape=[cmotsz], initializer=tf.constant_initializer(0.0))
            )


    def call(self, inputs):
        """
        :param inputs: The inputs in the shape [B, T, H].
        :return: Combined result
        """
        mots = []
        expanded = tf.expand_dims(inputs, ParallelConv.DUMMY_AXIS)
        for W, b in zip(self.Ws, self.bs):
            conv = tf.nn.conv2d(expanded, W, strides=[1, 1, 1, 1], padding="SAME", name="CONV")
            activation = self.activation(tf.nn.bias_add(conv, b), "activation")
            mot = tf.reduce_max(activation, [ParallelConv.TIME_AXIS], keepdims=True)
            mots.append(mot)
        combine = tf.reshape(tf.concat(values=mots, axis=ParallelConv.FEATURE_AXIS), [-1, self.output_dim])
        return combine

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    @property
    def requires_length(self):
        return False


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    U = tf.random.uniform(tf.shape(logits), 0, 1)
    gumbel_samples = -tf.math.log(-tf.math.log(U + eps) + eps)
    y = tf.nn.softmax((logits + gumbel_samples) / tau)
    if hard:
        y_hard = tf.cast(y == tf.reduce_max(y, -1, keepdims=True), tf.float32)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def lstm_cell(hsz: int, forget_bias: float = 1.0, **kwargs):
    """Produce a single cell with no dropout
    :param hsz: (``int``) The number of hidden units per LSTM
    :param forget_bias: (``int``) Defaults to 1
    :return: a cell
    """
    num_proj = kwargs.get("projsz")
    if num_proj and num_proj == hsz:
        num_proj = None
    cell = tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True, num_proj=num_proj)
    skip_conn = bool(kwargs.get("skip_conn", False))
    return tf.nn.rnn_cell.ResidualWrapper(cell) if skip_conn else cell


def lstm_cell_w_dropout(
    hsz: int, pdrop: float, forget_bias: float = 1.0, variational: bool = False, training: bool = False, **kwargs
):
    """Produce a single cell with dropout
    :param hsz: (``int``) The number of hidden units per LSTM
    :param pdrop: (``int``) The probability of keeping a unit value during dropout
    :param forget_bias: (``int``) Defaults to 1
    :param variational (``bool``) variational recurrence is on
    :param training (``bool``) are we training? (defaults to ``False``)
    :return: a cell
    """
    output_keep_prob = tf.contrib.framework.smart_cond(training, lambda: 1.0 - pdrop, lambda: 1.0)
    state_keep_prob = tf.contrib.framework.smart_cond(
        training, lambda: 1.0 - pdrop if variational else 1.0, lambda: 1.0
    )
    num_proj = kwargs.get("projsz")
    cell = tf.contrib.rnn.LSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True, num_proj=num_proj)
    skip_conn = bool(kwargs.get("skip_conn", False))
    cell = tf.nn.rnn_cell.ResidualWrapper(cell) if skip_conn else cell
    output = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=output_keep_prob,
        state_keep_prob=state_keep_prob,
        variational_recurrent=variational,
        dtype=tf.float32,
    )
    return output


class LSTMEncoder2(tf.keras.layers.Layer):
    """The LSTM encoder is a base for a set of encoders producing various outputs.

    All LSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)
    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input or `None`
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! Provide a graph layer name
        :param dropout_in_single_layer: TF only! If we have a single layer, should we dropout (defaults to `False`)
        :param skip_conn: TF only! This parameter isnt currently supported in TF Keras implementation
        :param projsz: TF only! This parameter isnt currently supported in TF Keras implementation
        """
        super().__init__(name=name)
        self.output_dim = hsz
        self._requires_length = requires_length
        self.rnns = []
        for _ in range(nlayers - 1):
            self.rnns.append(
                tf.keras.layers.LSTM(
                    hsz,
                    return_sequences=True,
                    recurrent_dropout=pdrop if variational else 0.0,
                    dropout=pdrop if not variational else 0.0,
                )
            )
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        self.rnns.append(
            tf.keras.layers.LSTM(
                hsz,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
        )

    def output_fn(self, output, state):
        return output, state

    def call(self, inputs):
        """RNNs over input sequence of `[B, T, C]` and lengths `[B]`, output `[B, S, H]` where `S = max(lengths)`

        :param inputs: A tuple of `(sequence, lengths)`, `sequence` shape `[B, T, C]`, lengths shape = `[B]`
        :return: Output depends on the subclass handling
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs
        rnnout, h, c = outputs
        return self.output_fn(rnnout, (h, c))

    @property
    def requires_length(self) -> bool:
        return self._requires_length


class LSTMEncoderWithState2(tf.keras.layers.Layer):

    """LSTM encoder producing the hidden state and the output, where the input doesnt require any padding
    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        self._requires_length = False
        self.hsz = hsz
        self.rnns = []
        for _ in range(nlayers - 1):
            self.rnns.append(
                tf.keras.layers.LSTM(
                    hsz,
                    return_sequences=True,
                    return_state=True,
                    recurrent_dropout=pdrop if variational else 0.0,
                    dropout=pdrop if not variational else 0.0,
                )
            )
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        self.rnns.append(
            tf.keras.layers.LSTM(
                hsz,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
        )
        self.requires_state = True

    @property
    def output_dim(self):
        return self.hsz

    def call(self, inputs):
        """The format of the output here is

        output: `[B, T, H]`
        hidden: List[(h, c), (h, c), ...]`
        :param inputs:
        :return:
        """
        inputs, hidden_state_input = inputs

        hidden_outputs = []
        initial_state = None
        for i, rnn in enumerate(self.rnns):
            if hidden_state_input is not None:
                hidden_state = hidden_state_input[i]
                initial_state = (hidden_state[0], hidden_state[1])
            outputs, h, c = rnn(inputs, initial_state=initial_state)
            hidden_outputs.append((h, c))
            inputs = outputs
        return outputs, hidden_outputs

    def zero_state(self, batchsz: int):
        """Zero state for LSTM with batch size given

        :param batchsz: The batch size
        """
        num_rnns = len(self.rnns)
        zstate = []
        for rnn in self.rnns:
            zstate.append(
                #rnn.get_initial_state(tf.zeros(()))
                (tf.zeros((batchsz, self.hsz), dtype=np.float32), tf.zeros((batchsz, self.hsz), dtype=tf.float32))
            )

        return zstate


class LSTMEncoderSequence2(LSTMEncoder2):

    """LSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def output_fn(self, output, state):
        """Return sequence `[B, S, H]` where `S = max(lengths)`

        :param output: The sequence
        :param state: The hidden state
        :return: The sequence `[B, S, H]`
        """
        return output


class LSTMEncoderHidden2(LSTMEncoder2):

    """LSTM encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, output, state):
        """Get the last hidden layer

        :param output:
        :param state:
        :return: hidden unit tensor of shape `[B, H]`
        """
        return state[0]


class LSTMEncoderHiddenContext2(LSTMEncoder2):

    def output_fn(self, output, state):
        """Return last hidden state `(h, c)`

        :param output: The sequence
        :param state: The hidden state
        :return: The last hidden state `(h, c)`
        """
        return state


class GRUEncoder(tf.keras.layers.Layer):

    """GRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def __init__(
            self,
            insz: Optional[int],
            hsz: int,
            nlayers: int = 1,
            pdrop: float = 0.0,
            variational: bool = False,
            requires_length: bool = True,
            name: Optional[str] = None,
            dropout_in_single_layer: bool = False,
            **kwargs,
    ):
        """Produce a stack of GRUs with dropout performed on all but the last layer.
        :param insz: An optional input size for parity with other layer backends.  Can pass `None`
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: variational recurrence is on, defaults to `False`
        :param requires_length: Does the input require an input length (defaults to ``True``)
        :param name: TF only! Put a name in the graph for this layer. Optional, defaults to `None`
        :param dropout_in_single_layer: TF only! If there is a single layer, should we do dropout, defaults to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz

        for _ in range(nlayers - 1):
            self.rnns.append(
                tf.keras.layers.GRU(
                    hsz,
                    return_sequences=True,
                    recurrent_dropout=pdrop if variational else 0.0,
                    dropout=pdrop if not variational else 0.0,
                )
            )
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        self.rnns.append(
            tf.keras.layers.GRU(
                hsz,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
        )

    def output_fn(self, output, state):
        return output, state

    def call(self, inputs):
        """RNNs over input sequence of `[B, T, C]` and lengths `[B]`, output `[B, S, H]` where `S = max(lengths)`

        :param inputs: A tuple of `(sequence, lengths)`, `sequence` shape `[B, T, C]`, lengths shape = `[B]`
        :return: Output depends on the subclass handling
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs
        rnnout, h = outputs
        return self.output_fn(rnnout, h)

    @property
    def requires_length(self) -> bool:
        return self._requires_length


class GRUEncoderAll(tf.keras.layers.Layer):
    """GRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`
    """
    def __init__(
            self,
            insz: Optional[int],
            hsz: int,
            nlayers: int = 1,
            pdrop: float = 0.0,
            variational: bool = False,
            requires_length: bool = True,
            name: Optional[str] = None,
            dropout_in_single_layer=False,
            **kwargs,
    ):
        """Produce a stack of GRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per GRU
        :param nlayers: The number of layers of GRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz

        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(
                hsz,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
            self.rnns.append(rnn)
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(
            hsz,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
            )

        # This concat mode only works on the sequences, we still are getting 4 objects back for the state
        self.rnns.append(rnn)

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and a hidden vector `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        # (num_layers * num_directions, batch, hidden_size):
        hs = []
        for rnn in self.rnns:
            outputs, h = rnn(inputs, mask=mask)
            hs.append(h)
            inputs = outputs

        h = tf.stack(hs)
        return self.output_fn(outputs, h)

    @property
    def requires_length(self):
        return self._requires_length


class GRUEncoderSequence(GRUEncoder):

    """GRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """

    def output_fn(self, output, state):
        """Return sequence `(BxTxC)`

        :param output: The sequence
        :param state: The hidden state
        :return: The sequence `(BxTxC)`
        """
        return output


class GRUEncoderHidden(GRUEncoder):

    """BiGRU encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, output, state):
        """Return last hidden state `h`

        :param output: The sequence
        :param state: The hidden state
        :return: The last hidden state `(h, c)`
        """
        return state


class LSTMEncoder1(tf.keras.layers.Layer):
    """The LSTM encoder is a base for a set of encoders producing various outputs.

    All LSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`)
    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input or `None`
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! Provide a graph layer name
        :param dropout_in_single_layer: TF only! If we have a single layer, should we dropout (defaults to `False`)
        :param skip_conn: TF only! Should there be residual connections between RNNs
        :param projsz: TF only! Should we do `LSTMP` operation to a different output size
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.output_dim = hsz

        if variational or dropout_in_single_layer:
            self.rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(
                        hsz, pdrop, variational=variational, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz
                    )
                    for _ in range(nlayers)
                ],
                state_is_tuple=True,
            )
        else:
            self.rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz)
                    if i < nlayers - 1
                    else lstm_cell(hsz, skip_conn=skip_conn, projsz=projsz)
                    for i in range(nlayers)
                ],
                state_is_tuple=True,
            )

    def call(self, inputs):
        """RNNs over input sequence of `[B, T, C]` and lengths `[B]`, output `[B, S, H]` where `S = max(lengths)`

        :param inputs: A tuple of `(sequence, lengths)`, `sequence` shape `[B, T, C]`, lengths shape = `[B]`
        :return: Output depends on the subclass handling
        """
        inputs, lengths = tensor_and_lengths(inputs)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        ns = tf.contrib.framework.get_name_scope()
        with tf.name_scope(ns), tf.variable_scope(ns):
            rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, sequence_length=lengths, dtype=tf.float32)
        state = (hidden[-1].h, hidden[-1].c)
        return self.output_fn(rnnout, state)

    def output_fn(self, output, state):
        return output, state

    @property
    def requires_length(self):
        return self._requires_length


class LSTMEncoderSequence1(LSTMEncoder1):

    """LSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.
    """

    def output_fn(self, output, state):
        """Return sequence `[B, S, H]` where `S = max(lengths)`

        :param output: The sequence
        :param state: The hidden state
        :return: The sequence `[B, S, H]`
        """
        return output


class LSTMEncoderAllLegacy(LSTMEncoder1):
    """LSTM encoder that passes along the full output and hidden states for each layer

    *TF Note*: This module does not change the underlying TF output, we only support this to make it easier to use
    with existing TF primitives, especially in seq2seq
    """
    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` and a length `[B]`
        :return: An output tensor `[B, S, H]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        ns = tf.contrib.framework.get_name_scope()
        with tf.name_scope(ns), tf.variable_scope(ns):
            rnnout, encoder_state = tf.nn.dynamic_rnn(self.rnn, inputs, sequence_length=lengths, dtype=tf.float32)

        return self.output_fn(rnnout, encoder_state)

    def output_fn(self, output, hidden):
        return output, hidden


class LSTMEncoderAll1(LSTMEncoderAllLegacy):
    """LSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *TF Note*: This module reorganizes the underlying TF output, which means you must be careful if using
    this with another tf 1 `RNNCell` implementation
    """

    def output_fn(self, output, hidden):

        h = []
        c = []
        for i in range(len(hidden)):
            h.append(hidden[i].h)
            c.append(hidden[i].c)

        encoder_state = tf.stack(h), tf.stack(c)
        return output, encoder_state


class LSTMEncoderHidden1(LSTMEncoder1):

    """LSTM encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, output, state):
        """Get the last hidden layer

        :param output:
        :param state:
        :return: hidden unit tensor of shape `[B, H]`
        """
        return state[0]


# TODO: make this available from PyTorch or get rid of it
class LSTMEncoderHiddenContext1(LSTMEncoder1):

    def output_fn(self, output, state):
        return state


class LSTMEncoderWithState1(LSTMEncoder1):

    """LSTM encoder producing the hidden state and the output, where the input doesnt require any padding
    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = True,
        **kwargs,
    ):
        super().__init__(
            insz=insz,
            hsz=hsz,
            nlayers=nlayers,
            pdrop=pdrop,
            variational=variational,
            requires_length=False,
            name=name,
            dropout_in_single_layer=dropout_in_single_layer,
            **kwargs,
        )
        self.requires_state = True

    def zero_state(self, batchsz: int):
        """Zero state for LSTM with batch size given

        :param batchsz: The batch size
        """
        return self.rnn.zero_state(batchsz, tf.float32)

    def call(self, inputs):
        inputs, hidden = inputs
        ns = tf.contrib.framework.get_name_scope()
        with tf.name_scope(ns), tf.variable_scope(ns):
            rnnout, hidden = tf.nn.dynamic_rnn(self.rnn, inputs, initial_state=hidden, dtype=tf.float32)
        return rnnout, hidden  # (hidden[-1].h, hidden[-1].c)


class LSTMEncoderAll2(tf.keras.layers.Layer):
    """LSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *PyTorch note*: Takes a vector of shape `[B, T, C]` or `[B, C, T]`, depending on input specification
    of `batch_first`. Also note that in PyTorch, this defaults to `True`

    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        dropout_in_single_layer=False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input or `None`
        :param hsz: The number of hidden units per LSTM
        :param nlayers: The number of layers of LSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! Do variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! Provide a graph layer name
        :param dropout_in_single_layer: TF only! If we have a single layer, should we dropout (defaults to `False`)
        :param skip_conn: TF only! Not supported with tf.keras.layers implementation
        :param projsz: TF only! Not supported with tf.keras.layers implementation
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.output_dim = hsz
        self.rnns = []
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.LSTM(
                hsz,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
            self.rnns.append(rnn)
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.LSTM(
            hsz,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
        )

        # This concat mode only works on the sequences, we still are getting 4 objects back for the state
        self.rnns.append(rnn)

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` and a length `[B]`
        :return: An output tensor `[B, S, H]` , and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        hs = []
        cs = []
        for rnn in self.rnns:
            outputs, h, c = rnn(inputs, mask=mask)
            hs.append(h)
            cs.append(c)
            inputs = outputs

        h = tf.stack(hs)
        c = tf.stack(cs)
        return self.output_fn(outputs, (h, c))

    @property
    def requires_length(self):
        return self._requires_length


class BiLSTMEncoderAll2(tf.keras.layers.Layer):
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int = 1,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        :param skip_conn: TF 1 only!  Not supported on this implementation
        :param projsz: TF 1 only! Not supported on this implementation
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.output_dim = hsz
        self.rnns = []
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.LSTM(
                hsz // 2,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.LSTM(
            hsz // 2,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
        )

        # This concat mode only works on the sequences, we still are getting 4 objects back for the state
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode="concat"))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` and a length `[B]`
        :return: An output tensor `[B, S, H], and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        # (num_layers * num_directions, batch, hidden_size):
        hs = []
        cs = []
        for rnn in self.rnns:
            outputs, h1, c1, h2, c2 = rnn(inputs, mask=mask)
            h = tf.stack([h1, h2])
            c = tf.stack([c1, c2])
            hs.append(h)
            cs.append(c)
            inputs = outputs

        _, B, H = get_shape_as_list(h)
        h = tf.reshape(tf.stack(hs), [-1, B, H * 2])
        c = tf.reshape(tf.stack(cs), [-1, B, H * 2])
        return self.output_fn(outputs, (h, c))

    @property
    def requires_length(self) -> bool:
        return self._requires_length


class BiLSTMEncoder2(tf.keras.layers.Layer):
    """BiLSTM encoder base for a set of encoders producing various outputs.

    All BiLSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.
    """
    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        dropout_in_single_layer: bool = False,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        :param skip_conn: TF 1 only!  Not supported on this implementation
        :param projsz: TF 1 only! Not supported on this implementation
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.LSTM(
                hsz // 2,
                return_sequences=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
            )
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.LSTM(
            hsz // 2,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
        )

        # This concat mode only works on the sequences, we still are getting 4 objects back for the state
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode="concat"))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs

        rnnout, h_fwd, c_fwd, h_bwd, c_bwd = outputs
        return self.output_fn(rnnout, ((h_fwd, c_fwd), (h_bwd, c_bwd)))

    @property
    def requires_length(self):
        return self._requires_length


class BiLSTMEncoderSequence2(BiLSTMEncoder2):

    """BiLSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.
    """

    def output_fn(self, rnnout, state):
        return rnnout


class BiLSTMEncoderHidden2(BiLSTMEncoder2):
    """BiLSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, rnnout, state):
        return tf.concat([state[0][0], state[1][0]], axis=-1)


class BiLSTMEncoderHiddenContext2(BiLSTMEncoder2):

    def output_fn(self, rnnout, state):
        return tuple(tf.concat([state[0][i], state[1][i]], axis=-1) for i in range(2))


class BiGRUEncoder(tf.keras.layers.Layer):
    """BiGRU encoder base for a set of encoders producing various outputs.

    All BiGRU encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.
    """
    def __init__(
            self,
            insz: Optional[int],
            hsz: int,
            nlayers: int,
            pdrop: float = 0.0,
            variational: bool = False,
            requires_length: bool = True,
            name: Optional[str] = None,
            dropout_in_single_layer: bool = False,
            **kwargs,
    ):
        """Produce a stack of BiGRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz
        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(
                hsz // 2,
                return_sequences=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
                )
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(
            hsz // 2,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
            )

        # This concat mode only works on the sequences, we still are getting 2 objects back for the state
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode="concat"))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        for rnn in self.rnns:
            outputs = rnn(inputs, mask=mask)
            inputs = outputs

        rnnout, h_fwd, h_bwd = outputs
        return self.output_fn(rnnout, (h_fwd, h_bwd))

    @property
    def requires_length(self):
        return self._requires_length


class BiGRUEncoderSequence(BiGRUEncoder):

    """BiGRU encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.

    """
    def output_fn(self, rnnout, state):
        return rnnout


class BiGRUEncoderHidden(BiGRUEncoder):

    """BiGRU encoder that returns the top hidden state

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """

    def output_fn(self, _, state):
        return tf.concat([state[0], state[1]], axis=-1)


class BiGRUEncoderAll(tf.keras.layers.Layer):
    """BiGRU encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a hidden vector `[L, B, H]`
    """
    def __init__(
            self,
            insz: Optional[int],
            hsz: int,
            nlayers: int = 1,
            pdrop: float = 0.0,
            variational: bool = False,
            requires_length: bool = True,
            name: Optional[str] = None,
            dropout_in_single_layer: bool = False,
            **kwargs,
    ):
        """Produce a stack of BiGRUs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiGRU (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiGRUs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.rnns = []
        self.output_dim = hsz

        for _ in range(nlayers - 1):
            rnn = tf.keras.layers.GRU(
                hsz // 2,
                return_sequences=True,
                return_state=True,
                recurrent_dropout=pdrop if variational else 0.0,
                dropout=pdrop if not variational else 0.0,
                )
            self.rnns.append(tf.keras.layers.Bidirectional(rnn))
        if nlayers == 1 and not dropout_in_single_layer and not variational:
            pdrop = 0.0
        rnn = tf.keras.layers.GRU(
            hsz // 2,
            return_sequences=True,
            return_state=True,
            recurrent_dropout=pdrop if variational else 0.0,
            dropout=pdrop if not variational else 0.0,
            )

        # This concat mode only works on the sequences, we still are getting 4 objects back for the state
        self.rnns.append(tf.keras.layers.Bidirectional(rnn, merge_mode="concat"))

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        """
        :param inputs: A tuple containing the input tensor `[B, T, C]` or `[B, H, C]` and a length `[B]`
        :return: An output tensor `[B, S, H] or `[B, H, S]` , and a hidden vector `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        mask = tf.sequence_mask(lengths)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        # (num_layers * num_directions, batch, hidden_size):
        hs = []
        for rnn in self.rnns:
            outputs, h1, h2 = rnn(inputs, mask=mask)
            h = tf.stack([h1, h2])
            hs.append(h)
            inputs = outputs

        _, B, H = get_shape_as_list(h)
        h = tf.reshape(tf.stack(hs), [-1, B, H * 2])
        return self.output_fn(outputs, h)

    @property
    def requires_length(self) -> bool:
        return self._requires_length


class BiLSTMEncoder1(tf.keras.layers.Layer):
    """BiLSTM encoder base for a set of encoders producing various outputs.

    All BiLSTM encoders inheriting this class will trim the input to the max length given in the batch.  For example,
    if the input sequence is `[B, T, C]` and the `S = max(lengths)` then the resulting sequence, if produced, will
    be length `S` (or more precisely, `[B, S, H]`).  Because its bidirectional, half of the hidden units given in the
    constructor will be applied to the forward direction and half to the backward direction, and these will get
    concatenated.
    """

    def __init__(
        self,
        insz: Optional[int],
        hsz: int,
        nlayers: int,
        pdrop: float = 0.0,
        variational: bool = False,
        requires_length: bool = True,
        name: Optional[str] = None,
        skip_conn: bool = False,
        projsz: Optional[int] = None,
        **kwargs,
    ):
        """Produce a stack of LSTMs with dropout performed on all but the last layer.

        :param insz: The size of the input (or `None`)
        :param hsz: The number of hidden units per BiLSTM (`hsz//2` used for each direction and concatenated)
        :param nlayers: The number of layers of BiLSTMs to stack
        :param pdrop: The probability of dropping a unit value during dropout, defaults to 0
        :param variational: TF only! apply variational dropout
        :param requires_length: Does this encoder require an input length in its inputs (defaults to `True`)
        :param name: TF only! A name to give the layer in the graph
        :param dropout_in_single_layer: TF only! If its a single layer cell, should we do dropout?  Default to `False`
        :param skip_conn: TF 1 only!  Do residual connections between RNN layers
        :param projsz: TF 1 only! Do LSTMP operation to this output size
        """
        super().__init__(name=name)
        self._requires_length = requires_length
        self.layers = nlayers
        self.output_dim = hsz

        hsz = hsz // 2
        if variational:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(
                        hsz, pdrop, variational=variational, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz
                    )
                    for _ in range(nlayers)
                ],
                state_is_tuple=True,
            )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(
                        hsz, pdrop, variational=variational, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz
                    )
                    for _ in range(nlayers)
                ],
                state_is_tuple=True,
            )
        else:
            self.fwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz)
                    if i < nlayers - 1
                    else lstm_cell(hsz, skip_conn=skip_conn, projsz=projsz)
                    for i in range(nlayers)
                ],
                state_is_tuple=True,
            )
            self.bwd_rnn = tf.contrib.rnn.MultiRNNCell(
                [
                    lstm_cell_w_dropout(hsz, pdrop, training=TRAIN_FLAG(), skip_conn=skip_conn, projsz=projsz)
                    if i < nlayers - 1
                    else lstm_cell(hsz)
                    for i in range(nlayers)
                ],
                state_is_tuple=True,
            )

    def output_fn(self, rnnout, state):
        return rnnout, state

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        ns = tf.contrib.framework.get_name_scope()
        with tf.name_scope(ns), tf.variable_scope(ns):
            rnnout, (fwd_state, backward_state) = tf.nn.bidirectional_dynamic_rnn(
                self.fwd_rnn, self.bwd_rnn, inputs, sequence_length=lengths, dtype=tf.float32
            )
        rnnout = tf.concat(axis=2, values=rnnout)
        return self.output_fn(
            rnnout, ((fwd_state[-1].h, fwd_state[-1].c), (backward_state[-1].h, backward_state[-1].c))
        )

    @property
    def requires_length(self) -> bool:
        return self._requires_length


class BiLSTMEncoderAllLegacy(BiLSTMEncoder1):
    """BiLSTM encoder that passes along the full output and hidden states for each layer

    *TF Note*: This module reorganizes the underlying TF output, which means you must be careful if using
    this with another tf 1 `RNNCell` implementation
    """

    def call(self, inputs):
        """
        :param inputs: A 1tuple containing the input tensor `[B, T, C]` and a length `[B]`
        :return: An output tensor `[B, S, H], and tuple of hidden `[L, B, H]` and context `[L, B, H]`
        """
        inputs, lengths = tensor_and_lengths(inputs)
        max_length = tf.reduce_max(lengths)
        inputs = inputs[:, :max_length, :]
        ns = tf.contrib.framework.get_name_scope()
        with tf.name_scope(ns), tf.variable_scope(ns):
            rnnout, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                self.fwd_rnn, self.bwd_rnn, inputs, sequence_length=lengths, dtype=tf.float32
            )
        rnnout = tf.concat(axis=2, values=rnnout)

        return self.output_fn(rnnout, encoder_state)

    def output_fn(self, rnnout, encoder_state):
        fwd_state, bwd_state = encoder_state
        encoder_state = []
        for i in range(self.layers):
            h = tf.concat([fwd_state[i].h, bwd_state[i].h], -1)
            c = tf.concat([fwd_state[i].c, bwd_state[i].c], -1)
            encoder_state.append(tf.contrib.rnn.LSTMStateTuple(h=h, c=c))
        encoder_state = tuple(encoder_state)
        return rnnout, encoder_state


class BiLSTMEncoderAll1(BiLSTMEncoderAllLegacy):
    """BiLSTM encoder that passes along the full output and hidden states for each layer

    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]`

    This returns a 2-tuple of outputs `[B, S, H]` where `S = max(lengths)`, for the output vector sequence,
    and a tuple of hidden vector `[L, B, H]` and context vector `[L, B, H]`, respectively

    *TF Note*: This module reorganizes the underlying TF output, which means you must be careful if using
    this with another tf 1 `RNNCell` implementation
    """

    def output_fn(self, out, encoder_state):
        fwd_state, bwd_state = encoder_state
        hs = []
        cs = []
        for i in range(self.layers):
            h = tf.concat([fwd_state[i].h, bwd_state[i].h], -1)
            c = tf.concat([fwd_state[i].c, bwd_state[i].c], -1)
            hs.append(h)
            cs.append(c)
        encoder_state = (tf.stack(hs), tf.stack(cs))
        return out, encoder_state


class BiLSTMEncoderSequence1(BiLSTMEncoder1):

    """BiLSTM encoder to produce the transduced output sequence.

    Takes a tuple of tensor, shape `[B, T, C]` and a lengths of shape `[B]` and produce an output sequence of
    shape `[B, S, H]` where `S = max(lengths)`.  The lengths of the output sequence may differ from the input
    sequence if the `max(lengths)` given is shorter than `T` during execution.
    """
    def output_fn(self, rnnout, state):
        return rnnout


class BiLSTMEncoderHidden1(BiLSTMEncoder1):

    """BiLSTM encoder that returns the top hidden state


    Takes a tuple containing a tensor input of shape `[B, T, C]` and lengths of shape `[B]` and
    returns a hidden unit tensor of shape `[B, H]`
    """
    def output_fn(self, rnnout, state):
        return tf.concat([state[0][0], state[1][0]], axis=-1)


class BiLSTMEncoderHiddenContext1(BiLSTMEncoder1):

    def output_fn(self, rnnout, state):
        return state


if get_version(tf) < 2:
    LSTMEncoder = LSTMEncoder1
    LSTMEncoderSequence = LSTMEncoderSequence1
    LSTMEncoderWithState = LSTMEncoderWithState1
    LSTMEncoderHidden = LSTMEncoderHidden1
    LSTMEncoderHiddenContext = LSTMEncoderHiddenContext1
    LSTMEncoderAll = LSTMEncoderAll1
    BiLSTMEncoder = BiLSTMEncoder1
    BiLSTMEncoderSequence = BiLSTMEncoderSequence1
    BiLSTMEncoderHidden = BiLSTMEncoderHidden1
    BiLSTMEncoderHiddenContext = BiLSTMEncoderHiddenContext1
    BiLSTMEncoderAll = BiLSTMEncoderAll1
    from tensorflow.contrib.crf import crf_decode, crf_log_norm, crf_unary_score, crf_binary_score

    def crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params):
        """Computes the unnormalized score for a tag sequence.

        This is a patched version of the contrib
        where we dont do any length 1 sequence optimizations.  This was causing a very odd error
        where the true branch of smart_cond was being executed despite the predicate evaluating to 0.
        This probably makes it even slower than usual :(

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
              compute the unnormalized score.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
          transition_params: A [num_tags, num_tags] transition matrix.
        Returns:
          sequence_scores: A [batch_size] vector of unnormalized sequence scores.
        """

        # Compute the scores of the given tag sequence.
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(tag_indices, sequence_lengths, transition_params)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores


else:
    LSTMEncoder = LSTMEncoder2
    LSTMEncoderSequence = LSTMEncoderSequence2
    LSTMEncoderWithState = LSTMEncoderWithState2
    LSTMEncoderHidden = LSTMEncoderHidden2
    LSTMEncoderHiddenContext = LSTMEncoderHiddenContext2
    LSTMEncoderAll = LSTMEncoderAll2
    BiLSTMEncoder = BiLSTMEncoder2
    BiLSTMEncoderSequence = BiLSTMEncoderSequence2
    BiLSTMEncoderHidden = BiLSTMEncoderHidden2
    BiLSTMEncoderHiddenContext = BiLSTMEncoderHiddenContext2
    BiLSTMEncoderAll = BiLSTMEncoderAll2
    from tensorflow_addons.text.crf import crf_decode, crf_sequence_score, crf_log_norm



class Reduction(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        pass


class ConcatReduction(Reduction):
    def __init__(self, output_dims: List[int], axis=-1):
        super().__init__()
        self.axis = axis
        self.output_dim = sum(output_dims)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        return tf.concat(values=inputs, axis=-1)


class ConcatSubtractReduction(Reduction):
    """This reduction assumes paired input and subtracts the two to get a distance

       It is useful for training sentence encoders and is used, for example, in SentenceBERT
       For this to work we assume that the inputs are paired, and subtract them
       """

    def __init__(self, output_dims: List[int], axis=-1):
        super().__init__()
        self.axis = axis
        self.output_dim = 3 * output_dims[0]

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        sub = tf.abs(inputs[0] - inputs[1])
        return tf.concat(values=[inputs[0], inputs[1], sub], axis=-1)


class SumReduction(Reduction):
    def __init__(self, output_dims: List[int]):
        super().__init__()
        # We could actually project if we needed, or at least should validate
        self.output_dim = output_dims[0]

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        return tf.add_n(inputs)


class SumLayerNormReduction(Reduction):

    def __init__(self, output_dims: List[int], layer_norm_eps: float = 1.0e-12):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.output_dim = output_dims[0]

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        outputs = tf.add_n(inputs)
        return self.ln(outputs)


class EmbeddingsStack(tf.keras.layers.Layer):
    def __init__(
        self,
        embeddings_dict: Dict[str, tf.keras.layers.Layer],
        dropout_rate: float = 0.0,
        requires_length: bool = False,
        reduction: Optional[Union[str, tf.keras.layers.Layer]] = 'concat',
        name: Optional[str] = None,
        **kwargs,
    ):
        """Takes in a dictionary where the keys are the input tensor names, and the values are the embeddings

        :param embeddings_dict: (``dict``) dictionary of each feature embedding
        """

        super().__init__(name=name)
        self.embeddings = embeddings_dict

        output_dims = []
        for embedding in embeddings_dict.values():
            output_dims += [embedding.get_dsz()]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self._requires_length = requires_length
        if isinstance(reduction, str):
            if reduction == 'sum':
                self.reduction = SumReduction(output_dims)
            elif reduction == 'sum-layer-norm':
                self.reduction = SumLayerNormReduction(output_dims, layer_norm_eps=kwargs.get('layer_norm_eps', 1.0e-12))
            else:
                self.reduction = ConcatReduction(output_dims)
        else:
            self.reduction = reduction
        self.dsz = self.reduction.output_dim

    def keys(self):
        return self.embeddings.keys()

    def items(self):
        return self.embeddings.items()

    def __getitem__(self, item: str):
        return self.embeddings[item]

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        i = 0
        for k, embedding in self.embeddings.items():
            x = inputs[k]
            # Its a hair faster to do this than using isinstance
            if x.__class__ == tuple:
                embeddings_out = embedding(*x)
            else:
                embeddings_out = embedding(x)
            all_embeddings_out.append(embeddings_out)
            i += 1
        word_embeddings = self.reduction(all_embeddings_out)
        return self.dropout(word_embeddings, TRAIN_FLAG())

    #@property
    #def dsz(self) -> int:
    #    total_dsz = 0
    #    for embeddings in self.embeddings.values():
    #        total_dsz += embeddings.get_dsz()
    #    return total_dsz

    def keys(self):
        return self.embeddings.keys()

    @property
    def requires_length(self) -> bool:
        return self._requires_length

    @property
    def output_dim(self) -> bool:
        return self.dsz


class WeightTieDense(tf.keras.layers.Layer):
    def __init__(self, tied, name="weight-tied", use_bias=False):
        super().__init__(name=name)
        self.tied = tied
        self.use_bias = use_bias

    def _add_bias(self, W):
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[tf.shape(W)[0], ], initializer='zeros', regularizer=None,
                                        constraint=None, dtype=self.W.dtype, trainable=True)
        else:
            self.bias = None

    def build(self, input_shape):
        emb = getattr(self.tied, "embedding_layer", None)
        if emb is not None:
            self._add_bias(getattr(emb, "W"))
            super().build(input_shape)
            return
        W = getattr(self.tied, "W", None)
        if W is not None:
            self._add_bias(W)
            super().build(input_shape)
            return
        self._add_bias(self.W)
        super().build()

    @property
    def W(self):
        w = getattr(self.tied, "W", None)
        return w if w is not None else getattr(self.tied, "kernel")

    def call(self, inputs):
        shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [-1, shape[-1]])
        outs = tf.matmul(inputs, self.W, transpose_b=True)
        if self.use_bias:
            outs = tf.nn.bias_add(outs, self.bias)
        new_shape = tf.concat([shape[:-1], tf.constant([-1])], axis=0)
        return tf.reshape(outs, new_shape)


class DenseStack(tf.keras.layers.Layer):
    def __init__(
        self,
        insz: Optional[int],
        hsz: Union[int, List[int]],
        activation: Union[tf.keras.layers.Activation, str, List[str]] = "relu",
        pdrop_value: float = 0.5,
        init: Optional[Any] = None,
        name: Optional[str] = None,
        skip_connect = False,
        layer_norm = False,
        **kwargs,
    ):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param hsz: The number of hidden units
        :param activation: The name of the activation function to use
        :param pdrop_value: The dropout probability
        :param init: The tensorflow initializer

        """
        super().__init__(name=name)
        hszs = listify(hsz)
        self.output_dim = hszs[-1]
        activations = listify(activation)
        if len(activations) == 1:
            activations = activations * len(hszs)
        if len(activations) != len(hszs):
            raise ValueError("Number of activations must match number of hidden sizes in a stack!")
        if layer_norm:
            layer_norm_eps = kwargs.get('layer_norm_eps', 1e-6)
        if skip_connect:
            if not insz:
                raise ValueError("In order to use skip connection, insz must be provided in DenseStack!")
            current = insz

        layer_stack = []
        for hsz, activation in zip(hszs, activations):
            if skip_connect and current == hsz:
                layer_stack.append(SkipConnection(hsz, activation))
                current = hsz
            else:
                layer_stack.append(tf.keras.layers.Dense(hsz, activation))
            if layer_norm:
                layer_stack.append(tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps))
        self.layer_stack = layer_stack
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
    def requires_length(self) -> bool:
        return False


class WithDropout(tf.keras.layers.Layer):
    """This is a utility wrapper that applies dropout after executing the layer it wraps

    For variational dropout, we use `SpatialDropout1D` as described in:
    https://github.com/keras-team/keras/issues/7290
    """
    def __init__(self, layer: tf.keras.layers.Layer, pdrop: float = 0.5, variational: bool = False):
        super().__init__()
        self.layer = layer
        self.dropout = tf.keras.layers.SpatialDropout1D(pdrop) if variational else tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        return self.dropout(self.layer(inputs), TRAIN_FLAG())

    @property
    def output_dim(self) -> int:
        try:
            return self.layer.output_dim
        except:
            return self.layer.units

class WithDropoutOnFirst(tf.keras.layers.Layer):
    """Wrapper for any layer that surrounds it with dropout

    This exists primarily for the LSTMEncoderWithState to allow dropout on the output while
    passing back the hidden state

    For variational dropout, we use `SpatialDropout1D` as described in:
    https://github.com/keras-team/keras/issues/7290
    """
    def __init__(self, layer: tf.keras.layers.Layer, pdrop: float = 0.5, variational: bool = False):
        super().__init__()
        self.layer = layer
        self.dropout = tf.keras.layers.SpatialDropout1D(pdrop) if variational else tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        outputs = self.layer(inputs)
        return self.dropout(outputs[0], TRAIN_FLAG()), outputs[1]

    @property
    def output_dim(self) -> int:
        return self.layer.output_dim


class Highway(tf.keras.layers.Layer):
    def __init__(self, input_size: int, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.proj = tf.keras.layers.Dense(input_size, activation="relu")
        self.transform = tf.keras.layers.Dense(
            input_size, bias_initializer=tf.keras.initializers.Constant(value=-2.0), activation="sigmoid"
        )
        self.output_dim = input_size

    def call(self, inputs):
        proj_result = self.proj(inputs)
        proj_gate = self.transform(inputs)
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * inputs)
        return gated

    @property
    def requires_length(self):
        return False


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, layer: Optional[tf.keras.layers.Layer] = None, name: Optional[str] = None, **kwargs):
        super().__init__(name=name)
        self.layer = layer

    def call(self, inputs):
        return inputs + self.layer(inputs)

    @property
    def requires_length(self) -> bool:
        return False


class SkipConnection(ResidualBlock):
    def __init__(self, input_size: int, activation: str = "relu"):
        super().__init__(tf.keras.layers.Dense(input_size, activation=activation))


class TimeDistributedProjection(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name=None):
        """Set up a low-order projection (embedding) by flattening the batch and time dims and matmul

        TODO: Avoid where possible, Dense should work in most cases

        :param name: The name for this scope
        :param num_outputs: The number of feature maps out
        """
        super().__init__(True, name)
        self.output_dim = num_outputs
        self.W = None
        self.b = None

    def build(self, input_shape):

        nx = int(input_shape[-1])
        self.W = self.add_weight("W", [nx, self.output_dim])
        self.b = self.add_weight("b", [self.output_dim], initializer=tf.constant_initializer(0.0))
        super().build(input_shape)

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
    def requires_length(self) -> bool:
        return False


class SequenceSequenceAttention(tf.keras.layers.Layer):
    def __init__(self, hsz: Optional[int] = None, pdrop: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.hsz = hsz
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.attn = None

    def call(self, qkvm):
        query, key, value, mask = qkvm
        a = self._attention(query, key, mask)
        self.attn = a
        a = self.dropout(a, training=TRAIN_FLAG())
        return self._update(a, value)

    def _attention(self, queries, keys, mask=None):
        pass

    def _update(self, a, value):
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T_q, T_k]
        :param values: The values [B, H, T_k, D]
        :returns: A tensor of shape [B, H, T_q, D]
        """
        return tf.matmul(a, value)


class SeqScaledDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop: float = 0.1, name: str = "scaled_dot_product_attention", **kwargs):
        super().__init__(pdrop, name=name, **kwargs)

    def _attention(self, query, key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: A tensor that is (BxHxTxT)
        """
        # Check why this was set to 2 before
        d_k = tf.shape(query)[-1]
        scores = tf.matmul(query, key, transpose_b=True)
        scores *= tf.math.rsqrt(tf.cast(d_k, tf.float32))

        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1e9)

        return tf.nn.softmax(scores, name="attention_weights")


class SequenceSequenceRelativeAttention(tf.keras.layers.Layer):
    """This form of attention is specified in Shaw et al 2018: https://www.aclweb.org/anthology/N18-2074.pdf
    """

    def __init__(self, hsz: int = None, pdrop: float = 0.1, name=None, **kwargs):
        super().__init__(name=name)
        self.hsz = hsz
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.attn = None

    def call(self, q_k_v_ek_ev_m):
        query, key, value, edges_key, edges_value, mask = q_k_v_ek_ev_m
        a = self._attention(query, key, edges_key, mask)
        self.attn = a
        a = self.dropout(a, training=TRAIN_FLAG())
        return self._update(a, value, edges_value)

    def _attention(self, query, key, edges_key, mask=None):
        pass

    def _update(self, a, value, edges_value):
        """Attention weights are applied for each value, but in a series of efficient matrix operations.

        In the case of self-attention, the key and query (used to create the attention weights)
        and values are all low order projections of the same input.

        :param a: The attention weights [B, H, T_q, T_k]
        :param value: The values [B, H, T_k, D]
        :param edge_value: The edge values [T_q, T_k, D]
        :returns: A tensor of shape [B, H, T_q, D]
        """
        B, H, T_k, D = get_shape_as_list(value)
        updated_values = tf.matmul(a, value)  # [B, H, T_q, D]
        if edges_value is not None:
            # (T, BxH, T)
            a = tf.transpose(tf.reshape(a, [B * H, -1, T_k]), [1, 0, 2])  # [T_q, BxH, T_k]
            t = tf.matmul(a, edges_value)  # (T_q, BxH, D)
            t = tf.transpose(t, [1, 0, 2])
            update_edge_values = tf.reshape(t, [B, H, -1, D])
            return updated_values + update_edge_values
        else:
            return updated_values


class SeqScaledDotProductRelativeAttention(SequenceSequenceRelativeAttention):
    def __init__(self, pdrop: float = 0.1, name: str = "scaled_dot_product_rel_attention", **kwargs):
        super().__init__(pdrop=pdrop, name=name, **kwargs)

    def _attention(self, query, key, edges_key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [T_q x T_k x D]
        :return: A tensor that is (B x H x T_q x T_k)
        """
        B, H, T_q, d_k = get_shape_as_list(query)
        scores_qk = tf.matmul(query, key, transpose_b=True)  # (., H, T_q, T_k) = (., H, T_q, D) x (., H, D, T_k)

        tbhd = tf.transpose(tf.reshape(query, [B * H, T_q, d_k]), [1, 0, 2])  # [T_q, B*H, D]
        scores_qek = tf.matmul(tbhd, edges_key, transpose_b=True)  # [T_q, B*H, T_k]
        scores_qek = tf.transpose(scores_qek, [1, 0, 2])
        scores_qek = tf.reshape(scores_qek, [B, H, T_q, -1])  # [B, H, T_q, T_k]
        scores = (scores_qk + scores_qek) / math.sqrt(d_k)

        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1e9)

        return tf.nn.softmax(scores, name="rel_attention_weights")


class SeqDotProductRelativeAttention(SequenceSequenceRelativeAttention):
    def __init__(self, pdrop: float = 0.1, name: str = "dot_product_rel_attention", **kwargs):
        super().__init__(pdrop=pdrop, name=name, **kwargs)

    def _attention(self, query, key, edges_key, mask=None):
        """Scaled dot product attention, as defined in https://arxiv.org/abs/1706.03762

        We apply the query to the keys to receive our weights via softmax in a series of efficient
        matrix operations. In the case of self-attntion the key and query are all low order
        projections of the same input.

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :param edges_key: a matrix of relative embeddings between each word in a sequence [TxTxD]
        :return: A tensor that is (BxHxTxT)
        """
        B, H, T_q, d_k = get_shape_as_list(query)
        scores_qk = tf.matmul(query, key, transpose_b=True)  # (., H, T_q, T_k) = (., H, T_q, D) x (., H, D, T_k)

        tbhd = tf.transpose(tf.reshape(query, [B * H, T_q, d_k]), [1, 0, 2])  # [T_q, B*H, D]
        scores_qek = tf.matmul(tbhd, edges_key, transpose_b=True)  # [T_q, B*H, T_k]
        scores_qek = tf.transpose(scores_qek, [1, 0, 2])
        scores_qek = tf.reshape(scores_qek, [B, H, T_q, -1])  # [B, H, T_q, T_k]
        scores = scores_qk + scores_qek

        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1e9)

        return tf.nn.softmax(scores, name="rel_attention_weights")


def unfold_tensor(tensor, dim, window_sz):
    """Unfold a tensor by applying a sliding window on a certain dimension with step 1 and padding of 0's. The window
    dimension is added after the T dimension

    :param tensor: the tensor to be unfolded, with shape [d_1, d_2, ..., T, ..., d_n]
    :param dim: the dimension along which unfolding is applied
    :param window_sz: sliding window size, need to be an odd number

    :return: the unfolded tensor with shape [d_1, d_2, ..., T, window_sz, ..., d_n]
    """
    half_window = (window_sz - 1) // 2
    if dim < 0:
        dim = len(tensor.shape) + dim
    paddings = [[0, 0]] * len(tensor.shape)
    paddings[dim] = [half_window, half_window]
    padded = tf.pad(tensor, paddings)
    return tf.signal.frame(padded, frame_length=window_sz, frame_step=1, axis=dim)


class SeqScaledWindowedRelativeAttention(SequenceSequenceRelativeAttention):
    """This class implements windowed relative attention, i.e. preventing attention beyond rpr_k. For efficiency,
    _attention and _update are implemented in a different way."""
    def __init__(self, pdrop: float = 0.1, name: str = "scaled_windowed_rel_attention", **kwargs):
        super().__init__(pdrop=pdrop, name=name, **kwargs)

    def _unfold_mask(self, mask, batchsz, rpr_k):
        """Transform mask into the unfolded format."""
        window_sz = 2 * rpr_k + 1
        if mask.shape[2] > 1:  # mask is from a subsequent mask, with [1, 1, T, T] or [B, 1, T, T]
            raise Exception("Windowed relative attention cannot be used with mask of size TxT.")
        else:  # mask is a sequence mask [B, 1, 1, T]
            mask = unfold_tensor(mask, dim=-1, window_sz=window_sz)  # [B, 1, 1, T, W]
            return tf.squeeze(mask, axis=1)  # [B, 1, T, W]

    def _attention(
            self, query, key, rpr_key, mask = None):
        """Implementation of attention considering RA masking: using tf.extract_volume_patches() to create an extra
        dimension representing the sliding window. Then when applying matmul, Q, K, V share the same T dimension.
        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldn't
        :param rpr_key: tensor of the rpr_key embeddings [W, d_k]
        :return: A tensor that is [B, H, T, 1, W] to be matmul with values
        """
        B, H, T, d_k = get_shape_as_list(query)
        window_sz = rpr_key.shape[0]
        rpr_k = (window_sz - 1) // 2
        query = tf.expand_dims(query, -2)  # [B, H, T, 1, d_k]
        key = unfold_tensor(key, dim=2, window_sz=window_sz)  # [B, H, T, W, d_k]
        rpr_key = tf.expand_dims(tf.expand_dims(tf.expand_dims(rpr_key, 0), 0), 0)  # [1, 1, 1, W, d_k]

        scores_qk = tf.matmul(query, key, transpose_b=True)  # [B, H, T, 1, W]
        scores_qrk = tf.matmul(query, rpr_key, transpose_b=True)  # [B, H, T, 1, W]
        scores = (scores_qk + scores_qrk) / math.sqrt(d_k)
        if mask is not None:
            mask = tf.expand_dims(self._unfold_mask(mask, B, rpr_k), -2)  # [B, 1, T, 1, W]
            scores = masked_fill(scores, tf.equal(mask, 0), -1e9)
        return tf.nn.softmax(scores, name="rel_attention_weights")

    def _update(self, a, value, rpr_value):
        # a has dim [B, H, T, 1, W]
        window_sz = a.shape[-1]
        value = unfold_tensor(value, dim=2, window_sz=window_sz)  # [B, H, T, W, d_value]
        updated_values = tf.matmul(a, value)  # [B, H, T, 1, d_value]
        if rpr_value is not None:
            rpr_value = tf.expand_dims(tf.expand_dims(tf.expand_dims(rpr_value, 0), 0), 0)  # [1, 1, 1, W, d_value]
            update_rpr_values = tf.matmul(a, rpr_value)  # [B, H, T, 1, d_value]
            return tf.squeeze(updated_values + update_rpr_values, axis=3)  # [B, H, T, d_value]
        else:
            return tf.squeeze(updated_values, axis=3)


class SeqDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop: float = 0.1, name: str = "dot_product_attention", **kwargs):
        super().__init__(pdrop, name=name, **kwargs)

    def _attention(self, query, key, mask=None):
        scores = tf.matmul(query, key, transpose_b=True)

        if mask is not None:
            scores = masked_fill(scores, tf.equal(mask, 0), -1e9)

        return tf.nn.softmax(scores, name="attention_weights")


class MultiHeadedAttention(tf.keras.layers.Layer):
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

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.1,
        scale: bool = False,
        d_k: Optional[int] = None,
        name: str = None,
    ):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super().__init__(name=name)

        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})")
        else:
            self.d_k = d_k

        self.h = num_heads
        if self.h > 1:
            self.d_value = self.d_k
        else:
            self.d_value = d_model
        self.w_Q = tf.keras.layers.Dense(units=self.d_k * self.h, name="query_projection")
        self.w_K = tf.keras.layers.Dense(units=self.d_k * self.h, name="key_projection")
        self.w_V = tf.keras.layers.Dense(units=self.d_value * self.h, name="value_projection")
        if self.h > 1:  # w_O is not needed for sinlge headed attention
            self.w_O = tf.keras.layers.Dense(units=self.d_k * self.h, name="output_projection")
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None

    def call(self, qkvm):
        query, key, value, mask = qkvm
        batchsz = get_shape_as_list(query)[0]

        # (B, T, H, D) -> (B, H, T, D)
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.w_V(value), [batchsz, -1, self.h, self.d_value]), [0, 2, 1, 3])
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn

        # (B, H, T, D) -> (B, T, H, D) -> (B, T, H*D)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batchsz, -1, self.h * self.d_value])
        if self.h > 1:
            return self.w_O(x)
        else:
            return x


class MultiHeadedRelativeAttention(tf.keras.layers.Layer):
    """
    Multi-headed relative attention from Shaw et al 2018 (https://www.aclweb.org/anthology/N18-2074.pdf)

    This method follows the same approach of MultiHeadedAttention, but it computes Relative Position Representations (RPR)
    which are used as part of the attention computations.  To facilitate this, the model has its own internal
    embeddings lookup table, and it has an updated computation for both the attention weights and the application
    of those weights to follow them.

    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        rpr_k: int,
        dropout: float = 0.1,
        scale: bool = False,
        d_k: Optional[int] = None,
        windowed_ra: bool = False,
        rpr_value_on: bool = True,
        name=None,
    ):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: Should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__()

        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})")
        else:
            self.d_k = d_k

        self.h = num_heads
        if self.h > 1:
            self.d_value = self.d_k
        else:
            self.d_value = d_model
        self.rpr_k = rpr_k
        self.rpr_value_on = rpr_value_on
        self.rpr_key = tf.keras.layers.Embedding(2 * rpr_k + 1, self.d_k)
        if self.rpr_value_on:
            self.rpr_value = tf.keras.layers.Embedding(2 * rpr_k + 1, self.d_value)
        self.windowed_ra = windowed_ra
        self.w_Q = tf.keras.layers.Dense(units=self.d_k * self.h, name="query_projection")
        self.w_K = tf.keras.layers.Dense(units=self.d_k * self.h, name="key_projection")
        self.w_V = tf.keras.layers.Dense(units=self.d_value * self.h, name="value_projection")
        if self.h > 1:  # w_O is not needed for sinlge headed attention
            self.w_O = tf.keras.layers.Dense(units=self.d_k * self.h, name="output_projection")
        if scale:
            if windowed_ra:
                self.attn_fn = SeqScaledWindowedRelativeAttention(dropout)
            else:
                self.attn_fn = SeqScaledDotProductRelativeAttention(dropout)
        else:
            self.attn_fn = SeqDotProductRelativeAttention(dropout)
        self.attn = None

    def make_rpr(self, q_len: int, k_len: int):
        """Create a matrix shifted by self.rpr_k and bounded between 0 and 2*self.rpr_k to provide 0-based indexing for embedding
        """
        q_seq = tf.range(q_len)
        k_seq = tf.range(k_len)
        window_len = 2 * self.rpr_k
        edges = tf.reshape(k_seq, [1, -1]) - tf.reshape(q_seq, [-1, 1]) + self.rpr_k  # [q_len, k_len]
        edges = tf.clip_by_value(edges, 0, window_len)
        if self.rpr_value_on:
            return self.rpr_key(edges), self.rpr_value(edges)  # [q_len, k_len, d_k]
        else:
            return self.rpr_key(edges), None

    def make_windowed_rpr(self):
        window_sz = 2 * self.rpr_k + 1
        window = tf.range(window_sz)
        if self.rpr_value_on:
            return self.rpr_key(window), self.rpr_value(window)
        else:
            return self.rpr_key(window), None

    def call(self, qkvm):
        """Low-order projections of query, key and value into multiple heads, then attention application and dropout

        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: Multi-head attention output, result of attention application to sequence (B, T, d_model)
        """
        query, key, value, mask = qkvm
        shp = get_shape_as_list(query)
        batchsz = shp[0]
        query_len = shp[1]
        key_len = get_shape_as_list(key)[1]  # key and value have the same length, but query can have a different length

        # (B, T, H, D) -> (B, H, T, D)
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(self.w_V(value), [batchsz, -1, self.h, self.d_value]), [0, 2, 1, 3])

        if self.windowed_ra:
            rpr_key, rpr_value = self.make_windowed_rpr()
        else:
            rpr_key, rpr_value = self.make_rpr(query_len, key_len)
        x = self.attn_fn((query, key, value, rpr_key, rpr_value, mask))
        self.attn = self.attn_fn.attn
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, H*D)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batchsz, -1, self.h * self.d_value])
        if self.h > 1:
            return self.w_O(x)
        else:
            return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[int] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: bool = False,
        rpr_value_on: bool = True,
        layer_drop: float = 0.0,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.layer_norms_after = layer_norms_after
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.layer_drop = layer_drop
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k,
                                                          windowed_ra=windowed_ra, rpr_value_on=rpr_value_on)
        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale=scale, d_k=d_k)

        self.ffn = FFN(d_model, activation_type, d_ff, ffn_pdrop, name="ffn")
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        """
        :param inputs: `(x, mask)`
        :return: The output tensor
        """
        x, mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        h = self.self_attn((x, x, x, mask))
        x = x + self.dropout(h, TRAIN_FLAG())
        x = self.ln2(x)
        x = x + self.dropout(self.ffn(x), TRAIN_FLAG())
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[int] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        layer_drop: float = 0.0,
        name: str = None
    ):
        super().__init__(name=name)
        self.d_model = d_model
        self.layer_norms_after = layer_norms_after
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        if rpr_k is not None:
            self.self_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k, name="self_attention")
            self.src_attn = MultiHeadedRelativeAttention(num_heads, d_model, rpr_k, pdrop, scale, d_k=d_k, name="src_attention")

        else:
            self.self_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k, name="self_attention")
            self.src_attn = MultiHeadedAttention(num_heads, d_model, pdrop, scale, d_k=d_k, name="src_attention")

        self.ffn = FFN(d_model, activation_type, d_ff, pdrop=ffn_pdrop, name="ffn")
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(pdrop)

    def call(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        if not self.layer_norms_after:
            x = self.ln1(x)
        x = x + self.dropout(self.self_attn((x, x, x, tgt_mask)), TRAIN_FLAG())

        x = self.ln2(x)
        x = x + self.dropout(self.src_attn((x, memory, memory, src_mask)), TRAIN_FLAG())

        x = self.ln3(x)
        x = x + self.dropout(self.ffn(x), TRAIN_FLAG())
        if self.layer_norms_after:
            x = self.ln1(x)
        return x


class TransformerEncoderStack(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: bool = False,
        rpr_value_on: bool = True,
        layer_drop: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):

        super().__init__(name=name)
        self.encoders = []
        self.ln = tf.identity if layer_norms_after else tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.layer_drop = layer_drop
        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers

        for i in range(layers):
            self.encoders.append(
                TransformerEncoder(
                    num_heads, d_model, pdrop, scale, activation, d_ff, d_k,
                    rpr_k=rpr_k[i], ffn_pdrop=ffn_pdrop,
                    layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps, windowed_ra=windowed_ra,
                    rpr_value_on=rpr_value_on, name=name,
                )
            )

    def call(self, inputs):
        x, mask = inputs
        for layer in self.encoders:
            ## TODO: FIXME I dont work on TPUs!!!
            ##pdrop = tf.random.uniform([])
            ##if not TRAIN_FLAG() or (pdrop >= self.layer_drop):
            x = layer((x, mask))
        return self.ln(x)


class TransformerEncoderStackWithLengths(TransformerEncoderStack):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: bool,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: Optional[bool] = False,
        rpr_value_on: bool = True,
        layer_drop: float = 0.0,
        name: str = None,
        **kwargs,
    ):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k,
                         ffn_pdrop, layer_norms_after, layer_norm_eps, windowed_ra, rpr_value_on, layer_drop, name, **kwargs)
        self.proj = WithDropout(tf.keras.layers.Dense(d_model), pdrop)

    def call(self, inputs):
        x, lengths = inputs
        x = self.proj(x)
        max_seqlen = get_shape_as_list(x)[1]
        mask = tf.expand_dims(tf.expand_dims(tf.sequence_mask(lengths, max_seqlen, dtype=tf.float32), 1), 1)
        return super().call((x, mask))


class TransformerEncoderStackWithTimeMask(TransformerEncoderStack):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: bool,
        scale: bool = True,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: Optional[float] = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        windowed_ra: Optional[bool] = False,
        rpr_value_on: bool = True,
        layer_drop: float = 0.0,
        name: str = None,
        **kwargs,
    ):
        super().__init__(num_heads, d_model, pdrop, scale, layers, activation, d_ff, d_k, rpr_k,
                         ffn_pdrop, layer_norms_after, layer_norm_eps, windowed_ra, rpr_value_on, layer_drop, name,
                         **kwargs)
        self.proj = WithDropout(tf.keras.layers.Dense(d_model), pdrop)

    def call(self, inputs):
        x, _ = inputs
        x = self.proj(x)
        max_seqlen = get_shape_as_list(x)[1]
        mask = subsequent_mask(max_seqlen)
        return super().call((x, mask))


class AttentionReduction(tf.keras.layers.Layer):
    """
    This is a reduction that is given Q, K, V and a mask vector.  Different from base reductions, which get an embedding stack
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, qkvm: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Inputs are the same as for a normal attention function, but the output here is a single tensor, ``[B, H]``
        :param query: a query for alignment. Can come from self in case of self-attn or decoder in case of E/D
        :param key: a set of keys from encoder or self
        :param value: a set of values from encoder or self
        :param mask: masking (for destination) to prevent seeing what we shouldnt
        :return: sentence-level encoding with dim [B, d_model]
        """


class SingleHeadReduction(AttentionReduction):
    """
    Implementation of the "self_attention_head" layer from the conveRT paper (https://arxiv.org/pdf/1911.03688.pdf)
    """
    def __init__(
            self, d_model: int, dropout: float = 0.0, scale: bool = False, d_k: Optional[int] = None, pooling: str = 'sqrt_length', name: Optional[str] = None,
    ):
        """
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param scale: should we scale the dot product attention
        :param d_k: The low-order project per head.  This is normally `d_model // num_heads` unless set explicitly
        """
        super().__init__(name=name)

        self.output_dim = d_model
        if d_k is None:
            self.d_k = d_model
        else:
            self.d_k = d_k
        self.w_Q = tf.keras.layers.Dense(self.d_k, name="query_projection")
        self.w_K = tf.keras.layers.Dense(self.d_k, name="key_projection")
        if scale:
            self.attn_fn = SeqScaledDotProductAttention(dropout)
        else:
            self.attn_fn = SeqDotProductAttention(dropout)
        self.attn = None
        pooling = pooling.lower()
        self.fill = 0
        if pooling == 'max':
            self.pool = self._max_pool
            self.fill = -1e9
        elif pooling == 'mean':
            self.pool = self._mean_pool
        else:
            self.pool = self._sqrt_length_pool

    def _sqrt_length_pool(self, x, seq_lengths):
        x = tf.reduce_sum(x, axis=1)  # [B, D]
        x = x * tf.expand_dims(tf.sqrt(tf.cast(seq_lengths, tf.float32)), -1)
        return x

    def _mean_pool(self, x, seq_lengths):
        return tf.reduce_sum(x, 1) / tf.expand_dims(seq_lengths, -1)

    def _max_pool(self, x, _):
        x = tf.reduce_max(x, 1)
        return x

    def call(self, qkvm: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:

        query, key, value, mask = qkvm
        batchsz = get_shape_as_list(query)[0]

        # (B, T, H, D) -> (B, H, T, D)
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, 1, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, 1, self.d_k]), [0, 2, 1, 3])
        value = tf.transpose(tf.reshape(value, [batchsz, -1, 1, self.output_dim]), [0, 2, 1, 3])
        x = self.attn_fn((query, key, value, mask))
        self.attn = self.attn_fn.attn
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, H*D)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batchsz, -1, self.output_dim])
        seq_lengths = tf.squeeze(tf.reduce_sum(tf.cast(mask != 0, tf.int32), axis=-1))


        pooled = self.pool(x, seq_lengths)
        return pooled

class TwoHeadConcat(AttentionReduction):
    """Use two parallel SingleHeadReduction, and concatenate the outputs. It is used in the conveRT
    paper (https://arxiv.org/pdf/1911.03688.pdf)"""

    def __init__(self, d_model, dropout, scale=False, d_k=None, pooling='sqrt_length', name=None):
        """Two parallel 1-head self-attention, then concatenate the output
        :param d_model: dim of the self-attention
        :param dropout: dropout of the self-attention
        :param scale: scale fo the self-attention
        :param d_k: d_k of the self-attention
        :return: concatenation of the two 1-head attention
        """
        super().__init__(name=name)
        self.output_dim = 2*d_model
        self.reduction1 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k, pooling=pooling)
        self.reduction2 = SingleHeadReduction(d_model, dropout, scale=scale, d_k=d_k, pooling=pooling)

    def call(self, inputs: tf.Tensor):
        x = inputs
        encoding1 = self.reduction1(x)
        encoding2 = self.reduction2(x)
        x = tf.concat([encoding1, encoding2], axis=-1)
        return x


class ConveRTFFN(tf.keras.layers.Layer):
    """Implementation of the FFN layer from the convert paper (https://arxiv.org/pdf/1911.03688.pdf)"""
    def __init__(self, insz, hszs, outsz, pdrop, name=None):
        """
        :param insz: input dim
        :param hszs: list of hidden sizes
        :param outsz: output dim
        :param pdrop: dropout of each hidden layer
        """
        super().__init__(name=name)
        self.dense_stack = DenseStack(insz,
                                      hszs,
                                      activation=gelu,
                                      pdrop_value=pdrop,
                                      skip_connect=True,
                                      layer_norm=True)
        self.final = tf.keras.layers.Dense(outsz)
        self.proj = tf.keras.layers.Dense(outsz) if insz != outsz else PassThru(insz)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.ln1(inputs)
        x = self.dense_stack(x)
        x = self.final(x)
        x = x + self.proj(inputs)
        return self.ln2(x)



class ContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, t=1.0, train_temperature=True):
        super().__init__()
        if t is None:
            t = 1.0
        self.t = self.add_weight("temperature", shape=[1], initializer=tf.constant_initializer(t), trainable=train_temperature)

    def call(self, model, inputs, targets):
        query = model.encode_query(inputs)  # [B, H]
        response = model.encode_response(targets)  # [B, H]
        query = tf.math.l2_normalize(query, axis=1)
        response = tf.math.l2_normalize(response, axis=1)
        B = get_shape_as_list(query)[0]
        labels = tf.range(0, B)
        logits = tf.matmul(query, response, transpose_b=True) * tf.exp(self.t)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss


class SymmetricContrastiveLoss(tf.keras.layers.Layer):
    def __init__(self, t=1.0, train_temperature=True):
        super().__init__()
        if t is None:
            t = 1.0
        self.t = self.add_weight("temperature", shape=[1], initializer=tf.constant_initializer(t),
                                 trainable=train_temperature)

    def call(self, model, inputs, targets):
        query = model.encode_query(inputs)  # [B, H]
        response = model.encode_response(targets)  # [B, H]
        query = tf.math.l2_normalize(query, axis=1)
        response = tf.math.l2_normalize(response, axis=1)
        B = get_shape_as_list(query)[0]
        labels = tf.range(0, B)
        logits = tf.matmul(query, response, transpose_b=True) * tf.exp(self.t)
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits, [1, 0]))
        loss_1 = tf.reduce_mean(loss_1)
        loss_2 = tf.reduce_mean(loss_2)

        loss = (loss_1 + loss_2) * 0.5
        return loss



class DualEncoderModel(tf.keras.layers.Layer):

    """Abstract base for dual encoders

    We can assume that our dual encoder needs to end up in the same output plane between the encoders, and we can define
    the set of losses here that we are likely to need for most.


    """
    def __init__(self, in_sz: int, stacking_layers: Union[int, List[int]] = None, d_out: int = 512,
                 ffn_pdrop=0.1, in_sz_2=None, name=None):
        super().__init__(name=name)

        if not in_sz_2:
            in_sz_2 = in_sz
        if stacking_layers:
            stacking_layers = listify(stacking_layers)
        if stacking_layers:
            self.ff1 = ConveRTFFN(in_sz, stacking_layers, d_out, ffn_pdrop)
            self.ff2 = ConveRTFFN(in_sz_2, stacking_layers, d_out, ffn_pdrop)
        elif in_sz != d_out or in_sz != in_sz_2:
            self.ff1 = tf.keras.layers.Dense(d_out)
            self.ff2 = tf.keras.layers.Dense(d_out)
        else:
            self.ff1 = PassThru(in_sz)
            self.ff2 = PassThru(in_sz)
        self.output_dim = d_out

    def encode_query_base(self, query: tf.Tensor) -> tf.Tensor:
        pass

    def encode_response_base(self, response: tf.Tensor) -> tf.Tensor:
        pass

    def encode_query(self, query: tf.Tensor) -> tf.Tensor:
        tensor = self.encode_query_base(query)
        return self.ff1(tensor)

    def encode_response(self, response: tf.Tensor) -> tf.Tensor:
        tensor = self.encode_response_base(response)
        return self.ff2(tensor)

    def call(self, inputs):
        query, response = inputs
        encoded_query = self.encode_query(query)
        encoded_response = self.encode_response(response)
        return encoded_query, encoded_response

    def create_loss(self, loss_type='symmetric', init_temp=None, learn_temp=False):
        if loss_type == 'contrastive':
            return ContrastiveLoss(init_temp, learn_temp)
        elif loss_type == 'symmetric':
            return SymmetricContrastiveLoss(init_temp, learn_temp)
        raise Exception(f"Unknown loss function type {loss_type}")


class PairedModel(DualEncoderModel):
    """Legacy model for transformer-based dual encoder

    This is a dual-encoder transformer model which shares the lower layer encoder transformer sub-graph
    The reduction layer is attention based and takes the same input as the transformer layers.  It pools the reprs
    Finally, the feed-forward stacks are applied via subclassing.

    Note that this model predates the more abstract `AbstractDualEncoder` which could accomplish the same thing
    by injecting the same `nn.Module` for encoder_1 and encoder_2 consisting of the transformer and reduction
    """
    def __init__(self, embeddings,
                 d_model,
                 d_ff,
                 dropout,
                 num_heads,
                 num_layers,
                 stacking_layers=None,
                 d_out=512,
                 d_k=None,
                 weight_std=0.02,
                 rpr_k=None,
                 reduction_d_k=64,
                 ffn_pdrop=0.1,
                 windowed_ra=False,
                 rpr_value_on=False,
                 reduction_type="2ha",
                 freeze_encoders=False,
                 name=None):
        super().__init__(2*d_model if reduction_type.startswith("2") else d_model, stacking_layers, d_out, ffn_pdrop, name=name)

        reduction_type = reduction_type.lower()
        if reduction_type == "2ha":
            self.reduction_layer = TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k)
        elif reduction_type == "2ha_mean":
            self.reduction_layer = TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k, pooling="mean")
        elif reduction_type == "2ha_max":
            self.reduction_layer = TwoHeadConcat(d_model, dropout, scale=False, d_k=reduction_d_k, pooling="max")
        elif reduction_type == "sha":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k)
        elif reduction_type == "sha_mean":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k, pooling="mean")
        elif reduction_type == "sha_max":
            self.reduction_layer = SingleHeadReduction(d_model, dropout, scale=False, d_k=reduction_d_k, pooling="max")
        else:
            raise Exception("Unknown exception type")

        self.embeddings = EmbeddingsStack({'x': embeddings})
        self.transformer = TransformerEncoderStack(num_heads=num_heads, d_model=d_model,
                                                   pdrop=dropout, layers=num_layers, activation='gelu', d_ff=d_ff,
                                                   ffn_pdrop=ffn_pdrop,
                                                   d_k=d_k, rpr_k=rpr_k, windowed_ra=windowed_ra, rpr_value_on=rpr_value_on)

        self.freeze = freeze_encoders

    def encode_query_base(self, query):
        query_mask = tf.cast(query != Offsets.PAD, tf.float32)
        att_mask = tf.expand_dims(tf.expand_dims(query_mask, 1), 1)

        embedded = self.embeddings({'x': query})
        encoded_query = self.transformer((embedded, att_mask))
        if self.freeze:
            encoded_query = tf.stop_gradient(encoded_query)

        encoded_query = self.reduction_layer((encoded_query, encoded_query, encoded_query, att_mask))
        return encoded_query

    def encode_response_base(self, response):
        response_mask = tf.cast(response != Offsets.PAD, tf.float32)
        att_mask = tf.expand_dims(tf.expand_dims(response_mask, 1), 1)
        embedded = self.embeddings({'x': response})
        encoded_response = self.transformer((embedded, att_mask))
        if self.freeze:
            encoded_response = tf.stop_gradient(encoded_response)
        encoded_response = self.reduction_layer((encoded_response, encoded_response, encoded_response, att_mask))
        return encoded_response



class TransformerDiscriminator(tf.keras.Model):
    """A Transformer model that tries to predict if each token is real or fake


    This model is based on [ELECTRA: Pre-Training Text Encoders as Discriminators Rather Than Generators,
    Clark et al. 2019](https://openreview.net/pdf?id=r1xMH1BtvB).

    """
    def __init__(
        self,
        embeddings,
        num_heads: int,
        d_model: int,
        dropout: bool,
        layers: int = 1,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.embeddings = EmbeddingsStack(embeddings, dropout)
        self.transformer = TransformerEncoderStack(
            num_heads, d_model=d_model, pdrop=dropout, scale=True,
            layers=layers, activation=activation, d_ff=d_ff, rpr_k=rpr_k, d_k=d_k,
            layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps
        )

        self.proj_to_output = tf.keras.layers.Dense(1)

        self.lengths_feature = kwargs.get('lengths_feature', list(self.embeddings.keys())[0])

    def call(self, features):
        embedded = self.embeddings(features)
        x = features[self.lengths_feature]
        input_mask = tf.expand_dims(tf.expand_dims(tf.cast(x != Offsets.PAD, tf.int32), 1), 1)
        #input_mask = tf.expand_dims(tf.expand_dims(tf.ones_like(x, dtype=tf.uint8), 1), 1)
        transformer_out = self.transformer((embedded, input_mask))
        binary = tf.squeeze(self.proj_to_output(transformer_out), -1)
        return binary

    def create_loss(self):

        def loss_fn(model, features, labels):
            logits = model(features)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
            return tf.reduce_mean(losses)
        return loss_fn


class TransformerDecoderStack(tf.keras.layers.Layer):

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        pdrop: float,
        scale: bool = True,
        layers: int = 1,
        activation_type: str = "relu",
        d_ff: Optional[int] = None,
        d_k: Optional[int] = None,
        rpr_k: Optional[Union[int, List[int]]] = None,
        ffn_pdrop: float = 0.0,
        layer_norms_after: bool = False,
        layer_norm_eps: float = 1.0e-6,
        layer_drop: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.decoders = []
        self.ln = tf.identity if layer_norms_after else tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.layer_drop = layer_drop
        if not is_sequence(rpr_k):
            rpr_k = [rpr_k] * layers

        for i in range(layers):
            self.decoders.append(
                TransformerDecoder(num_heads, d_model, pdrop, scale, activation_type, d_ff,
                                   d_k=d_k, rpr_k=rpr_k[i], ffn_pdrop=ffn_pdrop,
                                   layer_norms_after=layer_norms_after, layer_norm_eps=layer_norm_eps)
            )

    def call(self, inputs):
        x, memory, src_mask, tgt_mask = inputs
        for layer in self.decoders:
            ## TODO: FIXME I dont work on TPUs!!!
            ##pdrop = tf.random.uniform([])
            ##if not TRAIN_FLAG() or (pdrop >= self.layer_drop):
            x = layer((x, memory, src_mask, tgt_mask))
        return self.ln(x)


class FFN(tf.keras.layers.Layer):
    """
    FFN from https://arxiv.org/abs/1706.03762

    The paper does not specify any dropout in this layer, but subsequent implementations (like XLM) do use dropout.
    """

    def __init__(
        self,
        d_model: int,
        activation: str = "relu",
        d_ff: Optional[int] = None,
        pdrop: float = 0.0,
        name: Optional[int] = None,
    ):
        """Constructor, takes in model size (which is the external currency of each block) and the feed-forward size

        :param d_model: The model size.  This is the size passed through each block
        :param d_ff: The feed-forward internal size, which is typical 4x larger, used internally
        :param pdrop: The probability of dropping output
        """
        super().__init__(name=name)
        if d_ff is None:
            d_ff = 4 * d_model
        self.expansion = tf.keras.layers.Dense(d_ff)
        self.squeeze = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(pdrop)
        self.act = get_activation(activation)

    def call(self, inputs):
        return self.squeeze(self.dropout(self.act(self.expansion(inputs))))


class TaggerGreedyDecoder(tf.keras.layers.Layer):
    def __init__(self, num_tags: int, constraint_mask: Optional[Tuple[Any, Any]] = None, name: Optional[str] = None):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint_mask: Tuple[np.ndarray, np.ndarray], Constraints on the transitions [1, N, N]
        :param name: str, Optional name, defaults to `None`
        """
        super().__init__(name=name)
        self.num_tags = num_tags
        self.A = None
        self.inv_mask = None
        if constraint_mask is not None:
            _, inv_mask = constraint_mask
            self.inv_mask = inv_mask * -1e4

    def build(self, input_shape):
        self.A = self.add_weight(
            "transitions",
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        if self.inv_mask is not None:
            self.inv_mask = self.add_weight(
                "inverse_constraint_mask",
                shape=(self.num_tags, self.num_tags),
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.inv_mask),
                trainable=False
            )

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return tf.nn.log_softmax(self.A + self.inv_mask)
        return self.A

    def neg_log_loss(self, unary, tags, lengths):

        lengths = tf.cast(lengths, tf.int32)
        max_length = tf.reduce_max(lengths)
        tags = tags[:, :max_length]
        unary = unary[:, :max_length, :]
        mask = tf.sequence_mask(lengths, max_length)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tags, logits=unary)
        cross_entropy *= tf.cast(mask, tf.float32)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        return tf.reduce_mean(cross_entropy, name="loss")

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
            viterbi, path_scores = crf_decode(probv, self.transitions, lengths + 1)
            return tf.identity(viterbi[:, 1:], name="best"), path_scores
        else:
            return tf.argmax(unary, 2, name="best"), None


class CRF(tf.keras.layers.Layer):
    def __init__(self, num_tags: int, constraint_mask: Optional[Tuple[Any, Any]] = None, name: Optional[str] = None):
        """Initialize the object.
        :param num_tags: int, The number of tags in your output (emission size)
        :param constraint_mask: Tuple[np.ndarray, np.ndarray], Constraints on the transitions [1, N, N]
        :param name: str, Optional name, defaults to `None`
        """
        super().__init__(name=name)

        self.num_tags = num_tags
        self.A = None
        self.mask = None
        self.inv_mask = None
        if constraint_mask is not None:
            mask, inv_mask = constraint_mask
            self.mask = mask
            self.inv_mask = inv_mask * -1e4

    def build(self, input_shape):
        self.A = self.add_weight(
            "transitions",
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )
        if self.mask is not None:
            self.mask = self.add_weight(
                "constraint_mask",
                shape=(self.num_tags, self.num_tags),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.constant_initializer(self.mask)
            )
        if self.inv_mask is not None:
            self.inv_mask = self.add_weight(
                "inverse_constraint_mask",
                shape=(self.num_tags, self.num_tags),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.constant_initializer(self.inv_mask)
            )

    @property
    def transitions(self):
        if self.inv_mask is not None:
            return (self.A * self.mask) + self.inv_mask
        return self.A

    def score_sentence(self, unary, tags, lengths):
        """Score a batch of sentences.

        :param unary: torch.FloatTensor: [B, T, N]
        :param tags: torch.LongTensor: [B, T]
        :param lengths: torch.LongTensor: [B]

        :return: torch.FloatTensor: [B]
        """
        return crf_sequence_score(unary, tf.cast(tags, tf.int32), tf.cast(lengths, tf.int32), self.transitions)

    def call(self, inputs, training=False):

        unary, lengths = inputs
        if training:
            return crf_log_norm(unary, lengths, self.transitions)
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

        viterbi, path_scores = crf_decode(probv, self.transitions, lengths + 1)
        return tf.identity(viterbi[:, 1:], name="best"), path_scores

    def neg_log_loss(self, unary, tags, lengths):
        """Neg Log Loss with a Batched CRF.

        :param unary: unary outputs of length `[B, S, N]`
        :param tags: tag truth values `[B, T]`
        :param lengths: tensor of shape `[B]`

        :return: Tensor of shape `[B]`
        """
        lengths = tf.cast(lengths, tf.int32)
        max_length = tf.reduce_max(lengths)
        fwd_score = self((unary, lengths), training=True)
        tags = tags[:, :max_length]
        gold_score = self.score_sentence(unary, tags, lengths)
        log_likelihood = gold_score - fwd_score
        return -tf.reduce_mean(log_likelihood)


class MeanPool1D(tf.keras.layers.Layer):
    def __init__(
        self,
        dsz: int,
        trainable: bool = False,
        name: Optional[str] = None,
        dtype: int = tf.float32,
        batch_first: bool = True,
        *args,
        **kwargs,
    ):
        """This is a layers the calculates the mean pooling in a length awareway.

           This was originally a wrapper around tf.keras.layers.GlobalAveragePooling1D()
           but that had problems because the mask didn't work then the dimension we
           are pooling over was variable length.

           looking here https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/pooling.py#L639

           We can see that the input shape is being gotten as a list where for the
           value of `input_shape[step_axis]` is `None` instead of getting the shape
           via `tf.shape`. This means that when they do the reshape the
           broadcast_shape is `[-1, None, 1]` which causes an error.
        """
        super().__init__(trainable, name, dtype)
        self.output_dim = dsz
        self.reduction_dim = 1 if batch_first else 0

    def call(self, inputs):
        tensor, lengths = tensor_and_lengths(inputs)
        # Regardless of whether the input is batch first or time first the result of the
        # sum is `[B, H]` so the lengths (which is `[B]`) should always be expanded with
        # `-1` to `[B, -1]` so that is broadcasts.
        return tf.reduce_sum(tensor, self.reduction_dim) / tf.cast(tf.expand_dims(lengths, -1), tf.float32)

    @property
    def requires_length(self):
        return True


class TagSequenceModel(tf.keras.Model):
    def __init__(
        self,
        nc: int,
        embeddings: tf.keras.layers.Layer,
        transducer: tf.keras.layers.Layer,
        decoder: Optional[tf.keras.layers.Layer] = None,
        name: str = None,
    ):
        super().__init__(name=name)
        if isinstance(embeddings, dict):
            self.embed_model = EmbeddingsStack(embeddings)
        else:
            assert isinstance(embeddings, EmbeddingsStack)
            self.embed_model = embeddings
        self.path_scores = None
        self.transducer_model = transducer
        self.proj_layer = TimeDistributedProjection(nc)
        decoder_model = CRF(nc) if decoder is None else decoder
        self.decoder_model = decoder_model

    def transduce(self, inputs):
        lengths = inputs.get("lengths")

        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        transduced = self.proj_layer(self.transducer_model(embedded))
        return transduced

    def decode(self, transduced, lengths):
        path, self.path_scores = self.decoder_model((transduced, lengths))
        return path

    def call(self, inputs, training=None):
        transduced = self.transduce(inputs)
        return self.decode(transduced, inputs.get("lengths"))

    def neg_log_loss(self, unary, tags, lengths):
        return self.decoder_model.neg_log_loss(unary, tags, lengths)


class LangSequenceModel(tf.keras.Model):
    def __init__(
        self,
        nc: int,
        embeddings: tf.keras.layers.Layer,
        transducer: tf.keras.layers.Layer,
        decoder: Optional[tf.keras.layers.Layer] = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.embed_model = embeddings
        self.transducer_model = transducer
        if hasattr(transducer, "requires_state") and transducer.requires_state:
            self._call = self._call_with_state
            self.requires_state = True
        else:
            self._call = self._call_without_state
            self.requires_state = False
        self.output_layer = TimeDistributedProjection(nc)
        self.decoder_model = decoder

    def call(self, inputs):
        return self._call(inputs)

    def _call_with_state(self, inputs):

        h = inputs.get("h")

        embedded = self.embed_model(inputs)
        transduced, hidden = self.transducer_model((embedded, h))
        transduced = self.output_layer(transduced)
        return transduced, hidden

    def _call_without_state(self, inputs):
        embedded = self.embed_model(inputs)
        transduced = self.transducer_model((embedded, None))
        transduced = self.output_layer(transduced)
        return transduced, None


class EmbedPoolStackModel(tf.keras.Model):
    def __init__(
        self,
        nc: int,
        embeddings: tf.keras.layers.Layer,
        pool_model: tf.keras.layers.Layer,
        stack_model: Optional[tf.keras.layers.Layer] = None,
        output_model: Optional[tf.keras.layers.Layer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.embed_model = embeddings
        self.pool_model = pool_model
        self.stack_model = stack_model
        self.output_layer = tf.keras.layers.Dense(nc) if output_model is None else output_model

    def call(self, inputs):
        lengths = inputs.get("lengths")
        embedded = self.embed_model(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled) if self.stack_model is not None else pooled
        return self.output_layer(stacked)


class FineTuneModel(tf.keras.Model):
    def __init__(self, nc: int, embeddings: tf.keras.layers.Layer, stack_model: Optional[tf.keras.layers.Layer] = None):
        super().__init__()
        self.finetuned = embeddings
        self.stack_model = stack_model
        self.output_layer = tf.keras.layers.Dense(nc)

    def call(self, inputs):
        base_layers = self.finetuned(inputs)
        stacked = self.stack_model(base_layers) if self.stack_model is not None else base_layers
        return self.output_layer(stacked)


class CompositeModel(tf.keras.Model):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self._requires_length = any(getattr(m, "requires_length", False) for m in self.models)
        # self.output_dim = sum(m.output_dim for m in self.models)

    def call(self, inputs, training=None, mask=None):
        inputs, lengths = tensor_and_lengths(inputs)
        pooled = []
        for m in self.models:
            if getattr(m, "requires_length", False):
                pooled.append(m((inputs, lengths)))
            else:
                pooled.append(m(inputs))
        return tf.concat(pooled, -1)

    @property
    def requires_length(self):
        return self._requires_length


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


def skip_conns(inputs, wsz_all, n, activation_fn="relu"):
    x = inputs
    for i in range(n):
        x = SkipConnection(wsz_all, activation_fn)(x)
    return x


def parallel_conv(input_, filtsz, dsz, motsz, activation_fn="relu"):
    return ParallelConv(dsz, motsz, filtsz, activation_fn)(input_)


def char_word_conv_embeddings(
    char_vec, filtsz, char_dsz, nfeats, activation_fn=tf.nn.tanh, gating=skip_conns, num_gates=1
):
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
    more config params that raw `tf.compat.v1.Session()` calls dont

    :return: A `tf.compat.v1.Session`
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)


def reload_checkpoint(sess: tf.compat.v1.Session, checkpoint: str, blocks_to_skip: List[str] = None):
    """
    Get the intersection of all non-output layers and declared vars in this graph and restore them
    :param sess: A tensorflow session to restore from
    :param checkpoint: checkpoint to read from
    :param blocks_to_skip: which variables of the graph to skip on reload
    :return: None
    """
    if not blocks_to_skip:
        blocks_to_skip = ['OptimizeLoss', 'output/']
    latest = tf.train.latest_checkpoint(checkpoint)
    if not latest:
        latest = checkpoint
    LOGGER.info("Reloading %s", latest)
    model_vars = set([t[0] for t in tf.train.list_variables(latest)])
    g = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    for block in blocks_to_skip:
        g = [v for v in g if not v.op.name.startswith(block)]
    g = [v for v in g if v.op.name in model_vars]
    LOGGER.info("Restoring %s", g)
    saver = tf.compat.v1.train.Saver(g)
    saver.restore(sess, latest)


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF Mask.
    Returns a mask with invalid moves as 0 and valid moves as 1.
    """
    mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx).T
    inv_mask = (mask == 0).astype(np.float32)
    return mask, inv_mask


def tie_weight(weight, tie_shape):
    """Higher order function to share weights between two layers.

    Tensorflow will take a custom_getter inside of a variable scope.
    This method creates a getter that looks for a match in shapes. If they match,
    The weights are transposed and shared.

    """
    def tie_getter(getter, name, *args, **kwargs):
        if kwargs['shape'] == tie_shape:
            return tf.transpose(weight)
        return getter("{}".format(name), *args, **kwargs)
    return tie_getter


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


def tf_device_wrapper(func):
    @wraps(func)
    def with_device(*args, **kwargs):
        device = kwargs.get("device", "default")
        if device == "cpu" and "sess" not in kwargs:
            g = tf.compat.v1.Graph()
            sess = tf.compat.v1.Session(
                graph=g, config=tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count={"CPU": 1, "GPU": 0})
            )
            kwargs["sess"] = sess
            return func(*args, **kwargs)
        return func(*args, **kwargs)

    return with_device


class VectorSequenceAttention(tf.keras.layers.Layer):
    def __init__(self, hsz):
        super().__init__()
        self.hsz = hsz
        self.W_c = tf.keras.layers.Dense(hsz, use_bias=False)

    def call(self, qkvm):
        query_t, keys_bth, values_bth, keys_mask = qkvm
        # Output(t) = B x H x 1
        # Keys = B x T x H
        # a = B x T x 1
        a = self._attention(query_t, keys_bth, keys_mask)
        attended = self._update(a, query_t, values_bth)

        return attended

    def _attention(self, query_t, keys_bth, keys_mask):
        pass

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H)
        B, H = get_shape_as_list(a)
        a = tf.reshape(a, [B, 1, H])
        c_t = tf.squeeze(a @ values_bth, 1)

        attended = tf.concat([c_t, query_t], -1)
        attended = tf.nn.tanh(self.W_c(attended))
        return attended


class LuongDotProductAttention(VectorSequenceAttention):
    def __init__(self, hsz: int):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(query_t, 2)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1e9)
        a = tf.nn.softmax(a, axis=-1)
        return a


class ScaledDotProductAttention(VectorSequenceAttention):
    def __init__(self, hsz: int):
        super().__init__(hsz)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(query_t, 2)
        a = a / math.sqrt(self.hsz)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1e9)
        a = tf.nn.softmax(a, axis=-1)
        return a


class LuongGeneralAttention(VectorSequenceAttention):
    def __init__(self, hsz: int):
        super().__init__(hsz)
        self.W_a = tf.keras.layers.Dense(self.hsz, use_bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        a = keys_bth @ tf.expand_dims(self.W_a(query_t), 2)
        a = tf.squeeze(a, -1)
        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1e9)
        a = tf.nn.softmax(a, axis=-1)
        return a


class BahdanauAttention(VectorSequenceAttention):
    def __init__(self, hsz: int):
        super().__init__(hsz)
        self.hsz = hsz
        self.W_a = tf.keras.layers.Dense(self.hsz, use_bias=False)
        self.E_a = tf.keras.layers.Dense(self.hsz, use_bias=False)
        self.v = tf.keras.layers.Dense(1, use_bias=False)

    def _attention(self, query_t, keys_bth, keys_mask):
        B, T, H = get_shape_as_list(keys_bth)
        q = tf.reshape(self.W_a(query_t), [B, 1, H])
        u = self.E_a(keys_bth)

        z = tf.nn.tanh(q + u)
        a = tf.squeeze(self.v(z), -1)

        if keys_mask is not None:
            masked_fill(a, tf.equal(keys_mask, 0), -1e9)
        a = tf.nn.softmax(a, axis=-1)
        return a

    def _update(self, a, query_t, values_bth):
        # a = B x T
        # Want to apply over context, scaled by a
        # (B x 1 x T) (B x T x H) = (B x 1 x H) -> (B x H)
        # context_vector shape after sum == (batch_size, hidden_size)

        B, T_k = get_shape_as_list(a)
        a = tf.reshape(a, [B, 1, T_k])
        c_t = tf.squeeze(a @ values_bth, 1)
        attended = tf.concat([c_t, query_t], -1)
        attended = self.W_c(attended)
        return attended


class BilinearAttention(tf.keras.layers.Layer):

    def __init__(self, in_hsz, out_hsz=1, bias_x=True, bias_y=True, name=None):
        super().__init__(True, name)

        self.in_hsz = in_hsz
        self.out_hsz = out_hsz
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.W = None

    def build(self, input_shape):
        self.W = self.add_weight(
                name=f"{self.name}/Weight",
                shape=(self.out_hsz, self.in_hsz + self.bias_x, self.in_hsz + self.bias_y),
                initializer=tf.constant_initializer(0.),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, x, y, mask):
        r"""
        Args:
            x: ``[B, T, H]``.
            y: ``[B, T, H]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """
        if self.bias_x:
            shp = get_shape_as_list(x)[:-1]
            ones = tf.ones(shp + [1])
            x = tf.concat([x, ones], -1)
        if self.bias_y:
            shp = get_shape_as_list(y)[:-1]
            ones = tf.ones(shp + [1])
            y = tf.concat([y, ones], -1)
        x = tf.expand_dims(x, 1)
        y = tf.expand_dims(y, 1)
        u = x @ self.W
        s = u @ tf.transpose(y, [0, 1, 3, 2])
        if self.out_hsz == 1:
            s = tf.squeeze(s, 1)
        s = masked_fill(s, tf.expand_dims(tf.equal(mask, False), 1), -1e9)
        return s

def subsequent_mask(size: int):
    b = tf.compat.v1.matrix_band_part(tf.ones([size, size]), -1, 0)
    m = tf.reshape(b, [1, 1, size, size])
    return m


def gnmt_length_penalty(lengths, alpha=0.8):
    """Calculate a length penalty from https://arxiv.org/pdf/1609.08144.pdf

    The paper states the penalty as (5 + |Y|)^a / (5 + 1)^a. This is implemented
    as ((5 + |Y|) / 6)^a for a (very) tiny performance boost

    :param lengths: `np.array`: [B, K] The lengths of the beams.
    :param alpha: `float`: A hyperparameter. See Table 2 for a search on this
        parameter.

    :returns:
        `torch.FloatTensor`: [B, K, 1] The penalties.
    """
    penalty = tf.constant(np.power(((5.0 + lengths) / 6.0), alpha))
    return tf.expand_dims(penalty, -1)


def no_length_penalty(lengths):
    """A dummy function that returns a no penalty (1)."""
    return tf.expand_dims(np.ones_like(lengths), -1)


def repeat_batch(t, K, dim=0):
    """Repeat a tensor while keeping the concept of a batch.

    :param t: `torch.Tensor`: The tensor to repeat.
    :param K: `int`: The number of times to repeat the tensor.
    :param dim: `int`: The dimension to repeat in. This should be the
        batch dimension.

    :returns: `torch.Tensor`: The repeated tensor. The new shape will be
        batch size * K at dim, the rest of the shapes will be the same.

    Example::

        >>> a = tf.constant(np.arange(10).view(2, -1))
        >>> a
	tensor([[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9]])
	>>> repeat_batch(a, 2)
	tensor([[0, 1, 2, 3, 4],
		[0, 1, 2, 3, 4],
		[5, 6, 7, 8, 9],
		[5, 6, 7, 8, 9]])
    """
    shape = get_shape_as_list(t)
    tiling = [1] * (len(shape) + 1)
    tiling[dim + 1] = K
    tiled = tf.tile(tf.expand_dims(t, dim + 1), tiling)
    old_bsz = shape[dim]
    new_bsz = old_bsz * K
    new_shape = list(shape[:dim]) + [new_bsz] + list(shape[dim + 1 :])
    return tf.reshape(tiled, new_shape)


def update_lengths(lengths, eoses, idx):
    """Update the length of a generated tensor based on the first EOS found.

    This is useful for a decoding situation where tokens after an EOS
    can be something other than EOS. This also makes sure that a second
    generated EOS doesn't affect the lengths.

    :param lengths: `torch.LongTensor`: The lengths where zero means an
        unfinished sequence.
    :param eoses:  `torch.ByteTensor`: A mask that has 1 for sequences that
        generated an EOS.
    :param idx: `int`: What value to fill the finished lengths with (normally
        the current decoding timestep).

    :returns: `torch.Tensor`: The updated lengths tensor (same shape and type).
    """
    # If a length is 0 it has never had a length set so it is eligible to have
    # this EOS be the length.
    updatable_lengths = lengths == 0
    # If this length can be updated AND this token is an eos
    lengths_mask = updatable_lengths & eoses
    return masked_fill(lengths, lengths_mask, idx)


class BeamSearchBase:
    def __init__(self, beam: int = 1, length_penalty=None, **kwargs):
        self.length_penalty = length_penalty if length_penalty else no_length_penalty
        self.K = beam

    def init(self, encoder_outputs):
        pass

    def step(self, paths, extra):
        pass

    def update(self, beams, extra):
        pass

    def __call__(self, encoder_outputs, **kwargs):
        """Perform batched Beam Search.

        Note:
            The paths and lengths generated do not include the <GO> token.

        :param encoder_outputs: `namedtuple` The outputs of the encoder class.
        :param init: `Callable(ecnoder_outputs: encoder_outputs, K: int)` -> Any: A
            callable that is called once at the start of the search to initialize
            things. This returns a blob that is passed to other callables.
        :param step: `Callable(paths: torch.LongTensor, extra) -> (probs: torch.FloatTensor, extra):
            A callable that is does a single decoding step. It returns the log
            probabilities over the vocabulary in the last dimension. It also returns
            any state the decoding process needs.
        :param update: `Callable(beams: torch.LongTensor, extra) -> extra:
            A callable that is called to edit the decoding state based on the selected
            best beams.
        :param length_penalty: `Callable(lengths: torch.LongTensor) -> torch.floatTensor
            A callable that generates a penalty based on the lengths. Lengths is
            [B, K] and the returned penalty should be [B, K, 1] (or [B, K, V] to
            have token based penalties?)

        :Keyword Arguments:
        * *beam* -- `int`: The number of beams to use.
        * *mxlen* -- `int`: The max number of steps to run the search for.

        :returns:
            tuple(preds: torch.LongTensor, lengths: torch.LongTensor, scores: torch.FloatTensor)
            preds: The predicted values: [B, K, max(lengths)]
            lengths: The length of each prediction [B, K]
            scores: The score of each path [B, K]
        """
        mxlen = kwargs.get("mxlen", 100)
        bsz = get_shape_as_list(encoder_outputs.output)[0]
        extra = self.init(encoder_outputs)
        paths = tf.fill((bsz, self.K, 1), Offsets.GO)
        # This tracks the log prob of each beam. This is distinct from score which
        # is based on the log prob and penalties.
        log_probs = tf.zeros((bsz, self.K))
        # Tracks the lengths of the beams, unfinished beams have a lengths of zero.
        lengths = tf.zeros((bsz, self.K), np.int32)

        for i in range(mxlen - 1):
            probs, extra = self.step(paths, extra)
            V = get_shape_as_list(probs)[-1]
            probs = tf.reshape(probs, (bsz, self.K, V))  # [B, K, V]
            if i > 0:
                # This mask is for all beams that are done.
                done_mask = lengths != 0  # [B, K, 1]
                done_mask = tf.expand_dims(done_mask, -1)
                # Can creating this mask be moved out of the loop? It never changes but we don't have V
                # This mask selects the EOS token
                eos_mask = tf.cast(
                    tf.zeros((1, 1, V)) + tf.reshape(tf.cast(tf.range(V) == Offsets.EOS, tf.float32), (1, 1, V)),
                    done_mask.dtype,
                )
                # eos_mask[:, :, Offsets.EOS] = 1
                # This mask selects the EOS token of only the beams that are done.
                mask = done_mask & eos_mask
                # Put all probability mass on the EOS token for finished beams.
                # Otherwise as the other beams get longer they will all give
                # up and eventually select this beam and all outputs become
                # the same.
                probs = masked_fill(probs, done_mask, -1e8)
                probs = masked_fill(probs, mask, 0)
                probs = tf.expand_dims(log_probs, -1) + probs  # [B, K, V]
                # Calculate the score of the beam based on the current length.
                valid_lengths = masked_fill(lengths, lengths == 0, i + 1)
                path_scores = probs / tf.cast(self.length_penalty(valid_lengths), tf.float32)
            else:
                # On the first step we only look at probabilities for the first beam.
                # If we don't then the probs will be the same for each beam
                # This means the same token will be selected for each beam
                # And we won't get any diversity.
                # Using only the first beam ensures K different starting points.
                path_scores = probs[:, 0, :]

            flat_scores = tf.reshape(path_scores, (bsz, -1))  # [B, K * V]
            best_scores, best_idx = tf.math.top_k(flat_scores, self.K)
            # Get the log_probs of the best scoring beams
            probs = tf.reshape(probs, (bsz, -1))
            log_probs = gather_k(flat_scores, probs, best_idx, self.K)
            log_probs = tf.reshape(log_probs, (bsz, self.K))

            best_beams = best_idx // V  # Get which beam it came from
            best_idx = best_idx % V  # Get the index of the word regardless of which beam it is.

            # Best Beam index is relative within the batch (only [0, K)).
            # This makes the index global (e.g. best beams for the second
            # batch example is in [K, 2*K)).
            offsets = tf.range(bsz) * self.K
            offset_beams = best_beams + tf.expand_dims(offsets, -1)
            flat_beams = tf.reshape(offset_beams, [bsz * self.K])
            # Select the paths to extend based on the best beams
            flat_paths = tf.reshape(paths, [bsz * self.K, -1])
            new_paths = tf.gather(flat_paths, flat_beams)
            new_paths = tf.reshape(new_paths, [bsz, self.K, -1])
            # Add the selected outputs to the paths
            paths = tf.concat([new_paths, tf.expand_dims(best_idx, -1)], axis=2)

            # Select the lengths to keep tracking based on the valid beams left.
            ##
            flat_lengths = tf.reshape(lengths, [-1])
            lengths = tf.gather(flat_lengths, flat_beams)
            lengths = tf.reshape(lengths, (bsz, self.K))
            extra = self.update(flat_beams, extra)

            # Updated lengths based on if we hit EOS
            last = paths[:, :, -1]
            eoses = last == Offsets.EOS
            ##
            lengths = update_lengths(lengths, eoses, i + 1)
            if tf.reduce_sum(tf.cast(lengths != 0, np.int32)) == self.K:
                break
        else:
            # This runs if the loop didn't break meaning one beam hit the max len
            # Add an EOS to anything that hasn't hit the end. This makes the scores real.
            probs, extra = self.step(paths, extra)

            V = get_shape_as_list(probs)[-1]
            probs = tf.reshape(probs, (bsz, self.K, V))
            probs = probs[:, :, Offsets.EOS]  # Select the score of EOS
            # If any of the beams are done mask out the score of this EOS (they already had an EOS)
            probs = masked_fill(probs, (lengths != 0), 0)
            log_probs = log_probs + probs
            end_tokens = np.full((bsz, self.K, 1), Offsets.EOS)
            paths = tf.concat([paths, end_tokens], axis=2)
            lengths = update_lengths(lengths, np.ones_like(lengths) == 1, mxlen)
            best_scores = log_probs / tf.cast(tf.squeeze(self.length_penalty(lengths), -1), tf.float32)

        # Slice off the Offsets.GO token
        paths = paths[:, :, 1:]
        return paths, lengths, best_scores


class StackedLSTMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.rnn_size = rnn_size
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = []

        for i in range(num_layers):
            self.layers.append(tf.keras.layers.LSTMCell(rnn_size, use_bias=False))

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self) -> int:
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self.rnn_size

    def call(self, input, hidden):
        h_0, c_0 = hidden
        hs, cs = [], []
        for i, layer in enumerate(self.layers):
            input, (h_i, c_i) = layer(input, (h_0[i], c_0[i]))
            if i != self.num_layers - 1:
                input = self.dropout(input)
            hs.append(h_i)
            cs.append(c_i)

        hs = tf.stack(hs)
        cs = tf.stack(cs)

        return input, (hs, cs)


class StackedGRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_layers: int, input_size: int, rnn_size: int, dropout: float):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.layers = []

        for i in range(num_layers):
            self.layers.append(tf.keras.layers.GRUCell(rnn_size))

    def call(self, input, hidden):
        h_0 = hidden
        hs = []
        for i, layer in enumerate(self.layers):
            input, h_i = layer(input, h_0)
            if i != self.num_layers:
                input = self.dropout(input)
            hs.append(h_i)

        hs = tf.stack(hs)

        return input, hs

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self.rnn_size


if get_version(tf) < 2:

    def rnn_cell(hsz: int, rnntype: str, st: bool = None):
        """Produce a single RNN cell

        :param hsz: The number of hidden units per LSTM
        :param rnntype: `lstm` or `gru`
        :param st: state is tuple? defaults to `None`
        :return: a cell
        """
        if st is not None:
            cell = (
                tf.contrib.rnn.LSTMCell(hsz, state_is_tuple=st)
                if rnntype.endswith("lstm")
                else tf.contrib.rnn.GRUCell(hsz)
            )
        else:
            cell = tf.contrib.rnn.LSTMCell(hsz) if rnntype.endswith("lstm") else tf.contrib.rnn.GRUCell(hsz)
        return cell


else:

    def rnn_cell(insz: int, hsz: int, rnntype: str, nlayers: int = 1, dropout: float = 0.5):

        if rnntype == "gru":
            rnn = StackedGRUCell(nlayers, insz, hsz, dropout)
        else:
            rnn = StackedLSTMCell(nlayers, insz, hsz, dropout)
        return rnn
