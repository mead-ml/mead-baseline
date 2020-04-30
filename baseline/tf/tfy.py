import numpy as np
import tensorflow as tf
from baseline.utils import transition_mask as transition_mask_np, listify, read_json, is_sequence, import_user_module
from eight_mile.tf.layers import *
from functools import wraps

BaseLayer = tf.keras.layers.Layer
TensorDef = tf.Tensor

def reload_embeddings(embeddings_dict, basename):
    embeddings = {}
    for key, cls in embeddings_dict.items():
        embed_args = read_json('{}-{}-md.json'.format(basename, key))
        module = embed_args.pop('module')
        name = embed_args.pop('name', None)
        assert name is None or name == key
        mod = import_user_module(module)
        embed_args['name'] = key
        Constructor = getattr(mod, cls)
        embeddings[key] = Constructor(**embed_args)
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
    return mask, inv_mask


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


def stacked_dense(inputs, init, hszs=[], pdrop_value=0.5):
    return DenseStack(None, hszs, pdrop_value=pdrop_value, init=init)(inputs)

