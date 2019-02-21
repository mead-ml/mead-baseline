import numpy as np
import tensorflow as tf
from baseline.utils import transition_mask as transition_mask_np, listify, read_json, is_sequence, import_user_module
from eight_mile.tf.layers import *
from functools import wraps

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


def dense_layer(output_layer_depth):
    output_layer = tf.layers.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
    return output_layer


def reload_embeddings_from_state(embeddings_dict, basename):
    embeddings = {}
    for key, class_name in embeddings_dict.items():
        embed_args = read_json('{}-{}-md.json'.format(basename, key))
        module = embed_args.pop('module')
        name = embed_args.pop('name', None)
        assert name is None or name == key
        mod = import_user_module(module)
        Constructor = getattr(mod, class_name)
        embeddings[key] = Constructor(key, **embed_args)
    return embeddings


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
    return ParallelConvEncoderStack(get_shape_as_list(inputs)[-1], hsz, pdrop, nlayers, filts, activation_fn)(inputs, training)


def skip_conns(inputs, wsz_all, n, activation_fn='relu'):
    x = inputs
    for i in range(n):
        x = SkipConnection(wsz_all, activation_fn)(x)
    return x


def layer_norm(input, name, axis=[-1]):
    return LayerNorm(name=name, axis=axis)(input)


def lstm_encoder(embedseq, lengths, hsz, pdrop_value=0.5, variational=False, rnntype='blstm', layers=1):

    if rnntype == 'blstm':
        Encoder = BiLSTMEncoder
    else:
        Encoder = LSTMEncoder
    return Encoder(hsz, pdrop_value, layers, variational, rnn_signal)((embedseq, lengths), training=TRAIN_FLAG())


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

#def highway_conns(inputs, wsz_all, n):
#    x = inputs
#    for i in range(n):
#        x = Highway(wsz_all, name="highway-{}".format(i))(x)
#    return x

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


def pool_chars(x_char, Wch, ce0, char_dsz, nfeat_factor=None,
               cfiltsz=[3], max_feat=200, gating='skip',
               num_gates=1, activation_name='tanh', wsz=30):
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
                                                        activation_fn=get_activation(activation_name),
                                                        gating=gating_fn,
                                                        num_gates=num_gates)
            word_char = tf.reshape(cmot, [-1, mxlen, num_filts])

    return word_char, num_filts

def stacked_dense(inputs, init, hszs=[], pdrop_value=0.5):
    return DenseStack(hszs, pdrop_value=pdrop_value, init=init)(inputs)
