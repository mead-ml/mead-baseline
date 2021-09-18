import numpy as np
import tensorflow as tf
from baseline.utils import listify, read_json, is_sequence, import_user_module
from eight_mile.tf.layers import *
from functools import wraps
from eight_mile.tf.optz import EagerOptimizer
BaseLayer = tf.keras.layers.Layer
TensorDef = tf.Tensor


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


def dense_layer(output_layer_depth):
    output_layer = tf.layers.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
    return output_layer

def setup_tf2_checkpoints(optimizer: EagerOptimizer, model: tf.keras.layers.Layer, checkpoint_dir: str, max_to_keep: Optional[int] = 5) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
    """This sets up eager checkpointing, and restores existing checkpoints if they exist

    :param optimizer: The optimizer to connect
    :param model: The model to connect
    :param checkpoint_dir: The checkpoint directory, which may already exist or not
    :param max_to_keep: How many checkpoints to keep
    :return: The tf.train.Checkpoint, and the tf.train.CheckpointManager
    """
    _checkpoint = tf.train.Checkpoint(optimizer=optimizer.optimizer, model=model)

    checkpoint_manager = tf.train.CheckpointManager(_checkpoint,
                                                    directory=checkpoint_dir,
                                                    max_to_keep=max_to_keep)

    base = os.path.basename(checkpoint_dir)
    restore_file = None
    if base.startswith('ckpt-'):
        restore_file = checkpoint_dir
    elif os.path.isdir(checkpoint_dir):
        restore_file = checkpoint_manager.latest_checkpoint
    if restore_file:
        print(f'Restarting from: {restore_file}')
        _checkpoint.restore(restore_file)
    return _checkpoint, checkpoint_manager