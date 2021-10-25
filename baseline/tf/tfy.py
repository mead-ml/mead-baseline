import numpy as np
import tensorflow as tf
from eight_mile.utils import (
    listify, read_json, is_sequence
)
from baseline.utils import import_user_module
from eight_mile.tf.layers import *
from functools import wraps
from eight_mile.tf.optz import EagerOptimizer
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

