import os
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial, Offsets
from baseline.utils import transition_mask as transition_mask_np, listify
from baseline.tf.layers import *

import math


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


def get_basepath_or_cwd(model_file):
    """
    inspects the model_file variable for a directory name.

    if no directory is found, returns current working dir.
    """
    basepath = os.path.dirname(model_file)
    if not os.path.isdir(basepath):
        basepath = os.getcwd()

    return basepath


def dense_layer(output_layer_depth):
    output_layer = tf.keras.layers.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
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


def skip_conns_layers(inputs, wsz_all, n, activation_fn='relu'):
    x = inputs
    for i in range(n):
        x = SkipConnection(wsz_all, activation_fn)(x)
    return x


def skip_conns(inputs, wsz_all, n, activation_fn='relu', use_layers=True):
    """Produce one or more skip connection layers

    :param inputs: The sub-graph input
    :param wsz_all: The number of units
    :param n: How many layers of gating
    :return: graph output
    """
    if use_layers:
        return skip_conns_layers(inputs, wsz_all, n, activation_fn)
    activation_fn = tf_activation(activation_fn)
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = activation_fn(tf.matmul(inputs, W_p) + b_p, "skip_activation")

        inputs = inputs + proj
    return inputs


def highway_conns_layers(inputs, wsz_all, n):
    x = inputs
    for i in range(n):
        x = Highway(wsz_all)(x)
    return x


def highway_conns(inputs, wsz_all, n, use_layers=True):
    """Produce one or more highway connection layers

    :param inputs: The sub-graph input
    :param wsz_all: The number of units
    :param n: How many layers of gating
    :return: graph output
    """
    if use_layers:
        return highway_conns_layers(inputs, wsz_all, n)

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


def parallel_conv_layers(input_, filtsz, dsz, motsz, activation_fn='relu'):
    return ParallelConv(dsz, motsz, filtsz, activation_fn)(input_)


def parallel_conv(input_, filtsz, dsz, motsz, activation_fn='relu', use_layers=True):
    """Do parallel convolutions with multiple filter widths and max-over-time pooling.

    :param input_: The inputs in the shape [B, T, H].
    :param filtsz: The list of filter widths to use.
    :param dsz: The depths of the input (H).
    :param motsz: The number of conv filters to use (can be an int or a list to allow for various sized filters)
    :param activation_fn: The activation function to use (`default=tf.nn.relu`)
    :Keyword Arguments:
    * *activation_fn* -- (``callable``) The activation function to apply after the convolution and bias add
    """
    if use_layers:
        return parallel_conv_layers(input_, filtsz, dsz, motsz, activation_fn)
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
    return combine


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
               num_gates=1, activation='tanh', wsz=30):
    """Take in a tensor of characters (B x maxs x maxw) and do character convolution

    :param x_char: TF tensor for input characters, (B x maxs x maxw)
    :param Wch: A character embeddings matrix
    :param ce0: A control dependency for the embeddings that keeps the <PAD> value 0
    :param char_dsz: The character embedding dsz
    :param kwargs:

    :Keyword Arguments:
    * *cfiltsz* -- (``list``) A list of filters
    * *nfeat_factor* -- (``int``) A factor to be multiplied to filter size to decide number of hidden units
    * *max_feat* -- (``int``) The maximum number of hidden units per filter
    * *gating* -- (``str``) `skip` or `highway` supported, yielding residual conn or highway, respectively
    * *num_gates* -- (``int``) How many gating functions to apply
    * *activation* -- (``str``) A string name of an activation, (e.g. `tanh`)
    :return: The character compositional embedding and the number of hidden units as a tuple

    """
    filtsz = cfiltsz
    if nfeat_factor:
        max_feat = max_feat
        nfeats = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
    else:
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
