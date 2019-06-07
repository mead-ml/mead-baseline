import os
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial, Offsets, read_json, import_user_module
from baseline.utils import transition_mask as transition_mask_np
import math
from functools import wraps
from baseline.utils import is_sequence
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

def reload_embeddings(embeddings_dict, basename):
    embeddings = {}
    for key, cls in embeddings_dict.items():
        embed_args = read_json('{}-{}-md.json'.format(basename, key))
        module = embed_args.pop('module')
        name = embed_args.pop('name', None)
        assert name is None or name == key
        mod = import_user_module(module)
        Constructor = getattr(mod, cls)
        embeddings[key] = Constructor(key, **embed_args)
    return embeddings


def _add_ema(model, decay):
    """Create ops needed to track EMA when training.

    :param model: The model with a `.sess` we want to track.
    :param decay: float, Decay to use in the EMA

    :returns:
        ema_op: The update op. This applies the ema to each variable. Should be
           set as a control dependency on the training op.
        load: Op to copy emas to the variables.
        restore_var: Op to copy the original variables back from the EMA ones.

    Note:
        If you run the load op multiple times then the backup variables will be
        replaced by the ema variables.

        Currently there was a bug I haven't been able to fix. I haven't found why
        but sometimes when you run it with a tf.cond you get this error.
        `tensorflow.python.framework.errors_impl.InvalidArgumentError: Retval[0] does not have value`
        The stop gap is to remove this which means if you run load multiple times
        it will over write the backup variables with ema values.

        The load op is set up to automatically save the normal parameters when
        you load the ema's in.
    """
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    model_vars = model.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.variable_scope("BackupVariables"):
        backup_vars = [
            tf.get_variable(
                var.op.name,
                dtype=var.value().dtype,
                trainable=False,
                initializer=var.initialized_value()
            ) for var in model_vars
        ]
    ema_op = ema.apply(model_vars)

    save_back_up = tf.group(*(
        tf.assign(back, var.read_value())
        for var, back in zip(model_vars, backup_vars)
    ), name='save_backups')

    with tf.control_dependencies([save_back_up]):
        load = tf.group(*(
            tf.assign(var, ema.average(var).read_value())
            for var in model_vars
        ), name="load_ema")

    restore_vars = tf.group(*(
        tf.assign(var, back.read_value())
        for var, back in zip(model_vars, backup_vars)
    ), name="restore_backups")

    return ema_op, load, restore_vars


def transition_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF Mask.

    Returns a mask with invalid moves as 0 and valid moves as 1.
    """
    mask = transition_mask_np(vocab, span_type, s_idx, e_idx, pad_idx).T
    inv_mask = (mask == 0).astype(np.float32)
    return tf.constant(mask), tf.constant(inv_mask)


# TODO deprecated, remove
def _find_files_by_type(model_file, filetype):
    """Find all files by type, removing suffix

    we rely on the fact that vocab files end in .vocab.

    :return: the file names without the filetype.
    """
    matching_files = []

    filetype_ending = "." + filetype
    basepath = get_basepath_or_cwd(model_file)
    for filename in os.listdir(basepath):
        if filename.endswith(filetype_ending):
            filename_without_ending = filename[:-len(filetype_ending)]
            matching_files.append(os.path.join(basepath, filename_without_ending))

    if not matching_files:
        raise ValueError("no vocab files found in directory %s. \
Please specify the model as path-like. e.g. /data/model/model-name-1234" % basepath)

    return matching_files


def get_basepath_or_cwd(model_file):
    """
    inspects the model_file variable for a directory name.

    if no directory is found, returns current working dir.
    """
    basepath = os.path.dirname(model_file)
    if not os.path.isdir(basepath):
        basepath = os.getcwd()

    return basepath


# TODO: deprecated, remove this!
def get_vocab_file_suffixes(model_file):

    """Because our operations assume knowledge of the model name, we
    only need to return the suffix appended onto the end of the model
    name in the file.

    we make the assumption that a suffix is denoted with a hyphen.

    e.g.  a vocab file name = tagger-model-tf-30803-word.vocab
          would return ['word']

    :param model_file: the nonspecific path to the model. this could be
                /data/model/<model_name>. we need to remove the model name.
    :return:
    """
    filenames = _find_files_by_type(model_file, 'vocab')
    model_name = model_file.split('/')[-1]
    basepath = get_basepath_or_cwd(model_file)
    full_base = os.path.join(basepath, model_name)
    # the length of the name plus 1 for the hyphen separating the suffix.
    return [x[len(full_base)+1:] for x in filenames if x.startswith(full_base)]


def dense_layer(output_layer_depth):
    output_layer = layers_core.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
    return output_layer


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


def stacked_lstm(hsz, pdrop, nlayers, variational=False, training=False, **kwargs):
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
            [lstm_cell_w_dropout(hsz, pdrop, variational=variational, training=training, **kwargs) for _ in range(nlayers)],
            state_is_tuple=True
        )
    skip_conn = bool(kwargs.get('skip_conn', False))

    return tf.contrib.rnn.MultiRNNCell(
        [lstm_cell_w_dropout(hsz,
                             pdrop,
                             training=training,
                             skip_conn=False) if i < nlayers - 1 else lstm_cell(hsz,
                                                                                skip_conn=skip_conn) for i in range(nlayers)],
        state_is_tuple=True
    )


def stacked_cnn(inputs, hsz, pdrop, nlayers, filts=[5], activation_fn=tf.nn.relu, scope='StackedCNN', training=False):
    """Produce a stack of parallel or single convolution layers with residual connections and dropout between each

    :param inputs: The input
    :param hsz: (``int``) The number of hidden units per filter
    :param pdrop: (``float``) The probability of dropout
    :param nlayers: (``int``) The number of layers of parallel convolutions to stack
    :param filts: (``list``) A list of parallel filter widths to apply
    :param activation_fn: (``func``) A function for activation
    :param scope: A string name to scope this operation
    :return: a stacked CNN
    """
    with tf.variable_scope(scope):
        layers = []
        for filt in filts:
            # The first one cannot have a residual conn, since input size may differ
            layer = tf.layers.dropout(tf.layers.conv1d(inputs,
                                                   hsz,
                                                   filt,
                                                   activation=activation_fn,
                                                   padding="same",
                                                   name='conv{}-0'.format(filt)),
                                  pdrop, training=training,
                                  name='dropout{}-0'.format(filt))

            for i in range(1, nlayers):
                layer = layer + tf.layers.dropout(tf.layers.conv1d(inputs,
                                                               hsz,
                                                               filt,
                                                               activation=activation_fn,
                                                               padding="same",
                                                               name='conv{}-{}'.format(filt, i)),
                                              pdrop, training=training,
                                              name='dropout{}-{}'.format(filt, i))
            layers.append(layer)

        return tf.concat(values=layers, axis=2)


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

    
# # This function should never be used for decoding.  It exists only so that the training model can greedily decode
def show_examples_tf(model, es, rlut1, rlut2, vocab, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]
    i = 0
    src_lengths_key = model.src_lengths_key
    src_key = src_lengths_key.split('_')[0]
    while True:

        example = {}
        for k in batch_dict.keys():
            if i >= len(batch_dict[k]):
                return
            example[k] = batch_dict[k][i]
        print('========================================================================')

        src_i = example[src_key]
        src_len_i = example[src_lengths_key]
        tgt_i = example['tgt']

        sent = lookup_sentence(rlut1, src_i, reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        tgt_i = np.zeros((1, mxlen))
        example['tgt'] = tgt_i
        src_i = src_i[np.newaxis, :]
        example[src_key] = src_i
        example[src_lengths_key] = np.array([src_len_i])
        next_value = Offsets.GO
        for j in range(mxlen):
            tgt_i[0, j] = next_value
            tgt_len_i = np.array([j+1])
            example['tgt_lengths'] = tgt_len_i
            output = model.step(example).squeeze()[j]
            if sample is False:
                next_value = np.argmax(output)
            else:
                # This is going to zero out low prob. events so they are not
                # sampled from
                next_value = beam_multinomial(prob_clip, output)

            if next_value == Offsets.EOS:
                break

        sent = lookup_sentence(rlut2, tgt_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
        i += 1
        if i == max_examples:
            return


def skip_conns(inputs, wsz_all, n, activation_fn=tf.nn.relu):
    """Produce one or more skip connection layers

    :param inputs: The sub-graph input
    :param wsz_all: The number of units
    :param n: How many layers of gating
    :return: graph output
    """
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = activation_fn(tf.matmul(inputs, W_p) + b_p, "skip_activation")

        inputs = inputs + proj
    return inputs


def highway_conns(inputs, wsz_all, n):
    """Produce one or more highway connection layers

    :param inputs: The sub-graph input
    :param wsz_all: The number of units
    :param n: How many layers of gating
    :return: graph output
    """
    for i in range(n):
        with tf.variable_scope("highway-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu-proj")

            W_t = tf.get_variable("W_t", [wsz_all, wsz_all])
            b_t = tf.get_variable("B_t", [1, wsz_all], initializer=tf.constant_initializer(-2.0))
            transform = tf.nn.sigmoid(tf.matmul(inputs, W_t) + b_t, "sigmoid-transform")

        inputs = tf.multiply(transform, proj) + tf.multiply(inputs, 1 - transform)
    return inputs


def parallel_conv(input_, filtsz, dsz, motsz, activation_fn=tf.nn.relu):
    """Do parallel convolutions with multiple filter widths and max-over-time pooling.

    :param input_: The inputs in the shape [B, T, H].
    :param filtsz: The list of filter widths to use.
    :param dsz: The depths of the input (H).
    :param motsz: The number of conv filters to use (can be an int or a list to allow for various sized filters)
    :param activation_fn: The activation function to use (`default=tf.nn.relu`)
    :Keyword Arguments:
    * *activation_fn* -- (``callable``) The activation function to apply after the convolution and bias add
    """
    if not isinstance(motsz, list):
        motsz = [motsz] * len(filtsz)
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3
    expanded = tf.expand_dims(input_, DUMMY_AXIS)
    mots = []
    for fsz, cmotsz in zip(filtsz, motsz):
        with tf.variable_scope('cmot-%s' % fsz):
            kernel_shape = [1, fsz, dsz, cmotsz]
            W = tf.get_variable('W', kernel_shape)
            b = tf.get_variable(
                'b', [cmotsz],
                initializer=tf.constant_initializer(0.0)
            )
            conv = tf.nn.conv2d(
                expanded, W,
                strides=[1, 1, 1, 1],
                padding="SAME", name="CONV"
            )
            activation = activation_fn(tf.nn.bias_add(conv, b), 'activation')
            mot = tf.reduce_max(activation, [TIME_AXIS], keepdims=True)
            mots.append(mot)
    motsz_all = sum(motsz)
    combine = tf.reshape(tf.concat(values=mots, axis=FEATURE_AXIS), [-1, motsz_all])
    return combine, motsz_all


def time_distributed_projection(x, name, filters, w_init=None, b_init=tf.constant_initializer(0)):
    """Low-order projection (embedding) by flattening the batch and time dims and matmul

    :param x: The input tensor
    :param name: The name for this scope
    :param filters: The number of feature maps out
    :param w_init: An optional weight initializer
    :param b_init: An optional bias initializer
    :return:
    """
    with tf.variable_scope(name):
        shp = get_shape_as_list(x)
        nx = shp[-1]
        w = tf.get_variable("W", [nx, filters], initializer=w_init)
        b = tf.get_variable("b", [filters], initializer=b_init)
        collapse = tf.reshape(x, [-1, nx])
        c = tf.matmul(collapse, w)+b
        c = tf.reshape(c, shp[:-1] + [filters])
        return c


def gelu(x):
    return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))


def swish(x):
    return x*tf.nn.sigmoid(x)


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
    combine, wsz_all = parallel_conv(char_vec, filtsz, char_dsz, nfeats, activation_fn)
    joined = gating(combine, wsz_all, num_gates)
    return joined, wsz_all


def pool_chars(x_char, Wch, ce0, char_dsz, nfeat_factor=None, 
               cfiltsz=[3], max_feat=200, gating='skip',
               num_gates=1, activation='tanh', wsz=30):
    """Take in a tensor of characters (B x maxs x maxw) and do character convolution

    :param x_char: TF tensor for input characters, (B x maxs x maxw)
    :param Wch: A character embeddings matrix
    :param ce0: A control dependency for the embeddings that keeps the <PAD> value 0
    :param char_dsz: The character embedding dsz
    :param kwargs:

    :Keyword Arguments:
    * *cfiltsz* -- (``list``) A list of filters sizes, or a list of tuples of (filter size, num filts)
    * *nfeat_factor* -- (``int``) A factor to be multiplied to filter size to decide number of hidden units
    * *max_feat* -- (``int``) The maximum number of hidden units per filter
    * *gating* -- (``str``) `skip` or `highway` supported, yielding residual conn or highway, respectively
    * *num_gates* -- (``int``) How many gating functions to apply
    * *activation* -- (``str``) A string name of an activation, (e.g. `tanh`)
    :return: The character compositional embedding and the number of hidden units as a tuple

    """
    if is_sequence(cfiltsz[0]):
        filtsz = [filter_and_size[0] for filter_and_size in cfiltsz]
        nfeats = [filter_and_size[1] for filter_and_size in cfiltsz]

    elif nfeat_factor:
        max_feat = max_feat
        filtsz = cfiltsz
        nfeats = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
    else:
        filtsz = cfiltsz
        nfeats = wsz
    mxlen = tf.shape(x_char)[1]

    gating_fn = highway_conns if gating.startswith('highway') else skip_conns

    with tf.variable_scope("Chars2Word"):
        with tf.control_dependencies([ce0]):
            mxwlen = tf.shape(x_char)[-1]
            char_bt_x_w = tf.reshape(x_char, [-1, mxwlen])
            cembed = tf.nn.embedding_lookup(Wch, char_bt_x_w, name="embeddings")
            cmot, num_filts = char_word_conv_embeddings(cembed, filtsz, char_dsz, nfeats,
                                                        activation_fn=tf_activation(activation),
                                                        gating=gating_fn,
                                                        num_gates=num_gates)
            word_char = tf.reshape(cmot, [-1, mxlen, num_filts])

    return word_char, num_filts


def embed(x, vsz, dsz, initializer, finetune=True, scope="LUT"):
    """Perform a lookup table operation while freezing the PAD vector.  Use the initializer to set the weights

    :param x: The input to this operation
    :param vsz: The size of the input vocabulary
    :param dsz: The output size or embedding dimension
    :param initializer: An operation to initialize the weights
    :param finetune: Should the weights be fine-tuned during training or held constant?
    :param scope: A string scoping this operation
    :return: The sub-graph end
    """
    with tf.variable_scope(scope):
        W = tf.get_variable("W",
                            initializer=initializer,
                            shape=[vsz, dsz], trainable=finetune)
        e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
        with tf.control_dependencies([e0]):
            word_embeddings = tf.nn.embedding_lookup(W, x)

    return word_embeddings


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


def layer_norm(input, name, axis=[-1]):

    def _norm(x, g=None, b=None, e=1e-5, axis=[1]):
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + e)
        if g is not None and b is not None:
            x = x*g + b
        return x

    with tf.variable_scope(name):
        n_state = input.get_shape().as_list()[-1]
        gv = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        bv = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        return _norm(input, gv, bv, axis=axis)
