import logging
import time
import os
from argparse import ArgumentParser
import math
from typing import Tuple
import numpy as np
import baseline
from eight_mile.utils import str2bool, write_json, revlut, print_table
from collections import namedtuple
import baseline.tf.embeddings
import baseline.embeddings
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import Average, Timer, get_num_gpus_multiworker
from eight_mile.optz import *
from eight_mile.tf.optz import *
from eight_mile.tf.layers import get_shape_as_list, TransformerDiscriminator, SET_TRAIN_FLAG, create_distribute_strategy, read_yaml_tf
from baseline.tf.lm import TransformerMaskedLanguageModel
from eight_mile.tf.serialize import save_tlm_npz
import tensorflow as tf
import json
logger = logging.getLogger(__file__)

"""Pre-train an discriminator Transformer model in TensorFlow (supports TPUs)

This file uses Baseline to train a Transformer-based discriminative model
model, similar to (https://openreview.net/pdf?id=r1xMH1BtvB)

It is a stripped down version of `pretrain_discrim_pytorch.py` in the same directory, focused only
on handling preprocessed MLM data, and with no dynamic masking.  The assumptions are

* MLM preprocessed records are used as the starting point:
  - `x` is the masked signal for the MLM
  - `y` is the reconstructed signal for the MLM
* The MLM will attempt to learn to reconstruct the proper `y` value which is used for its loss
  - The MLM output softmax is sampled from in order to create a `noised_x` which is used for the discriminator
  - The `y_discrim` is dynamically determined as `1` if the reconstructed input from the MLM is the same as the `y`,
  `0` if its a "fake" and `1` everywhere else since these are all from the original signal
"""


def create_keras_optimizer(**kwargs):
    """Get an optimizer from kwargs

    Since this model requires training 2 different models, the usual EagerOptimizer object doesnt really
    fit our needs.  This function extracts the core logic (except for AdamW which is currently not supported)

    :param kwargs:
    :return:
    """
    if "lr_function" in kwargs:
        lr_function = kwargs["lr_function"]
    else:
        if "lr_scheduler_type" not in kwargs:
            kwargs["lr_scheduler_type"] = "default"
        lr_function = create_lr_scheduler(**kwargs)
    # decay_fn = None
    # Right now this option is pointless since sparse updates dont work on GPU.  We just turn it off
    sgd_mom = float(kwargs.get("mom", 0.9))
    clip = kwargs.get("clip", 100)


    optim = kwargs.get("optim", "sgd")
    lr = kwargs.get("lr", kwargs.get("eta", 0.01))

    if optim == "adadelta":
        rho = float(kwargs.get("rho", 0.95))
        eps = float(kwargs.get("epsilon", 1e-6))
        logger.info("adadelta(eta=%f, rho=%f, epsilon=%f)", lr, rho, eps)
        optimizer = tf.keras.optimizers.Adadelta(lr_function, rho, eps)
    elif optim == "adam":
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        eps = float(kwargs.get("epsilon", 1e-8))
        logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
        optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)
    elif optim == "rmsprop":
        # Get mom again with difference default
        mom = float(kwargs.get("mom", 0.0))
        logger.info("rmsprop(eta=%f, mom=%f)", lr, mom)
        optimizer = tf.keras.optimizers.RMSprop(lr_function, momentum=mom)
    elif sgd_mom > 0:
        logger.info("sgd-mom(eta=%f, mom=%f)", lr, sgd_mom)
        optimizer = tf.keras.optimizers.SGD(lr_function, sgd_mom)
    else:
        logger.info("sgd(eta=%f)", lr)
        optimizer = tf.keras.optimizers.SGD(lr_function)

    logger.info("clip gradients at %s", clip)
    return optimizer, clip


def mlm_loss(logits, labels):
    loss_mask = tf.cast(labels != 0, tf.float32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = losses * loss_mask
    losses = tf.reduce_sum(losses)
    non_zero = tf.reduce_sum(loss_mask)
    losses /= non_zero
    return losses


def discrim_loss_fn(logits, labels):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
    return tf.reduce_mean(losses)


def gen_vs_discrim(noised_x, labels, gen_model, discrim_model, mask_value):
    masked_indices = (noised_x == mask_value)

    logits = gen_model({'x': noised_x}, None)[0]
    gen_loss_step = mlm_loss(logits, labels)
    true_or_fake = 1 - tf.cast(masked_indices, tf.int64)
    recon_labels = best_from(logits) * tf.cast(masked_indices, tf.int64) + noised_x * true_or_fake
    # FIXME: the ELECTRA paper doesnt penalize when the MLM gets the right token
    logits = discrim_model({'x': recon_labels})

    discrim_loss_step = discrim_loss_fn(logits, true_or_fake)
    acc = get_accuracy(logits, true_or_fake)
    return gen_loss_step, discrim_loss_step, acc


def best_from(x_preds):
    B, T, V = get_shape_as_list(x_preds)
    sample_dist = tf.exp(tf.reshape(x_preds, (B * T, V)))
    output = tf.reshape(tf.random.categorical(sample_dist, num_samples=1), (B, T))
    return output


def get_accuracy(preds, true_or_fake):
    nz_preds = tf.reshape(preds, [-1])
    nz_true_or_fake = tf.cast(tf.reshape(true_or_fake, [-1]), tf.bool)

    preds_true = tf.squeeze(nz_preds > 0.5)
    num = tf.reduce_sum(tf.cast(nz_true_or_fake == preds_true, tf.int32))
    denom = nz_true_or_fake.shape.num_elements()
    return (num / denom)


def _parse_json(example):
    j = json.loads(example.numpy())
    return tf.constant(j['x'], dtype=tf.int32), tf.constant(j['y'], dtype=tf.int32)


feature_description = {
    'x': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    'y': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
}


def _parse_tf_record(example_proto):
    record = tf.io.parse_single_example(example_proto, feature_description)
    return record['x'], record['y']


def decode_json(example):
    return tf.py_function(_parse_json, [example], [tf.int32, tf.int32])


def get_dataset(directory, file_type, num_parallel_reads=1, shuffle=True):
    """Get a dataset as a tf.data.Dataset.  Input can be a bucket or a local file


    :param directory: Either a bucket or a file
    :param file_type: Currently supports "json" files or "tfrecords"
    :param num_parallel_reads: The number of parallel reads
    :param shuffle: Defaults to True
    :return: a `tf.data.Dataset`
    """
    pattern = os.path.join(directory, f'*.{file_type}')
    files = tf.io.gfile.glob(pattern)
    logger.debug(files)

    if file_type == 'json':
        ds = tf.data.TextLineDataset(files, num_parallel_reads=num_parallel_reads)
        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.map(decode_json)
        return ds
    if not shuffle:
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads)
    else:
        ds = tf.data.Dataset.from_tensor_slices(tf.constant(files))
        ds = ds.shuffle(buffer_size=len(files))
        ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           cycle_length=num_parallel_reads)
        ds = ds.shuffle(buffer_size=100)
    ds = ds.map(_parse_tf_record)
    return ds


def get_num_samples(sample_md):
    yml = read_yaml_tf(sample_md)
    return yml['num_samples']


def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_dir", type=str, required=True, help='Training directory')
    parser.add_argument("--valid_dir", type=str, required=True, help='Validation directory')
    parser.add_argument("--train_md", type=str, help="Training metadata YAML, defaults to `{train_dir}/md.yml`")
    parser.add_argument("--valid_md", type=str, help="Validation metadata YAML, defaults to `{valid_dir}/md.yml`")
    parser.add_argument("--dataset_key", default="tlm",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")

    parser.add_argument("--gen_d_model", type=int, default=256, help="Model dimension (and embedding dsz)")
    parser.add_argument("--gen_d_ff", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--gen_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--gen_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--gen_num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument('--gen_rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument('--windowed_ra', type=str2bool, default=False, help="whether prevent attention beyond rpr_k")
    parser.add_argument("--gen_loss_scale", type=float, default=50.0, help="Scaling for loss function")
    parser.add_argument("--gen_dropout", type=float, default=0.1, help="Dropout")

    parser.add_argument('--discrim_rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')

    parser.add_argument("--discrim_d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--discrim_d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--discrim_d_k", type=int, default=None, help="Dimension per head.  Use if num_heads=1 to reduce dims")
    parser.add_argument("--discrim_num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--discrim_num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--discrim_dropout", type=float, default=0.1, help="Dropout")

    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--distribute", type=str, default="mirror", choices=["mirror", "tpu", "nccl"])
    parser.add_argument("--tpu_ep", type=str, help="The TPU endpoint if using `distribute=tpu`")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--file_type", default='tfrecord', choices=['json', 'tfrecord'], help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--optim", default="adam", type=str, help="Optimizer to use (defaults to adam)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=32, help="Num training epochs")
    parser.add_argument("--restart", type=str2bool, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--causal", type=str2bool, default=False, help="Use CLM (causal) instead of MLM")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of checkpoints to save per epoch")
    parser.add_argument("--strategy", help="Training strategy, defaults to `mirror`", choices=["mirror"])
    parser.add_argument("--npz", help="Should we write out NPZ files?", type=str2bool, default=False)
    parser.add_argument("--tb", help="Turn on tensorboard?", type=str2bool, default=False)
    parser.add_argument("--convert_only", help="Should we just convert this file to NPZ and exit?", type=str2bool, default=False)
    args = parser.parse_args()
    SET_TRAIN_FLAG(True)

    if args.convert_only:
        args.restart = True
        args.npz = True

    if args.basedir is None:
        args.basedir = f'discrim-{args.dataset_key}-bpe-{os.getpid()}'
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Writing results to {args.basedir}")

    if args.tb:
        logdir = f"logs/scalars/{os.getpid()}"
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()
        logger.info(f"Set up tensorboard logdir {logdir}")

    strategy = create_distribute_strategy(args.distribute, args.tpu_ep)
    num_replicas = strategy.num_replicas_in_sync
    logger.info(f"Using {num_replicas} replicas in this job.")
    vectorizer = BPEVectorizer1D(model_file=args.subword_model_file, vocab_file=args.subword_vocab_file, mxlen=args.nctx)
    vocab = {'x': vectorizer.vocab}
    gen_preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.gen_d_model, known_vocab=vocab['x'],
                                                           preserve_vocab_indices=True,
                                                           embed_type=args.embed_type)

    vocabs = gen_preproc_data['vocab']

    discrim_preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.discrim_d_model, known_vocab=vocab['x'],
                                                               preserve_vocab_indices=True,
                                                               embed_type=args.embed_type)

    def dataset_train_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(args.batch_size)
        ds = get_dataset(args.train_dir, args.file_type, args.num_train_workers).batch(batch_size)
        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    train_loader = strategy.experimental_distribute_datasets_from_function(dataset_train_fn)

    def dataset_test_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(args.batch_size)
        ds = get_dataset(args.valid_dir, args.file_type, args.num_train_workers, shuffle=False).batch(batch_size)
        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    valid_loader = strategy.experimental_distribute_datasets_from_function(dataset_test_fn)

    train_md = args.train_md if args.train_md else os.path.join(args.train_dir, 'md.yml')
    num_train_samples = get_num_samples(train_md)
    valid_md = args.valid_md if args.valid_md else os.path.join(args.valid_dir, 'md.yml')
    num_valid_samples = get_num_samples(valid_md)
    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    gen_embeddings = {'x': gen_preproc_data['embeddings']}
    discrim_embeddings =  {'x': discrim_preproc_data['embeddings']}
    logger.info("Loaded embeddings")

    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)
    if len(args.gen_rpr_k) == 0 or args.gen_rpr_k[0] < 1:
        gen_rpr_k = None
    elif len(args.gen_rpr_k) == 1:
        gen_rpr_k = args.gen_rpr_k[0]
    else:
        gen_rpr_k = args.gen_rpr_k

    if len(args.discrim_rpr_k) == 0 or args.discrim_rpr_k[0] < 1:
        discrim_rpr_k = None
    elif len(args.gen_rpr_k) == 1:
        discrim_rpr_k = args.discrim_rpr_k[0]
    else:
        discrim_rpr_k = args.discrim_rpr_k

    gen_model = TransformerMaskedLanguageModel.create(gen_embeddings,
                                                      hsz=args.gen_d_model,
                                                      d_ff=args.gen_d_ff,
                                                      tie_weights=True,
                                                      dropout=args.gen_dropout,
                                                      gpu=False,
                                                      num_heads=args.gen_num_heads,
                                                      layers=args.gen_num_layers,
                                                      rpr_k=gen_rpr_k,
                                                      d_k=args.gen_d_k,
                                                      windowed_ra=args.windowed_ra,
                                                      src_keys=['x'], tgt_key='x')

    discrim_model = TransformerDiscriminator(discrim_embeddings, d_model=args.discrim_d_model, d_ff=args.discrim_d_ff,
                                             dropout=args.discrim_dropout,
                                             num_heads=args.discrim_num_heads, layers=args.discrim_num_layers,
                                             rpr_k=discrim_rpr_k, d_k=args.discrim_d_k)

    logger.info("Loaded model and loss")
    steps_per_epoch = num_train_samples // args.batch_size
    steps_per_valid_epoch = num_valid_samples // args.batch_size
    update_on = steps_per_epoch // args.saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")

    lr_decay = CosineDecaySchedulerTensorFlow(steps_per_epoch * args.epochs, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerTensorFlow(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRSchedulerTensorFlow(linear_warmup, lr_decay)

    mask_value = vocabs.get("[MASK]", vocabs.get("<MASK>", -1))
    if mask_value == -1:
        logger.error("We could not find a suitable masking token in the vocab")
        return

    optimizer, clip = create_keras_optimizer(**vars(args))

    discrim_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=discrim_model)
    discrim_checkpoint_manager = tf.train.CheckpointManager(discrim_checkpoint,
                                                            directory=os.path.join(args.basedir, 'discrim'),
                                                            max_to_keep=5)

    gen_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=discrim_model)
    gen_checkpoint_manager = tf.train.CheckpointManager(gen_checkpoint,
                                                        directory=os.path.join(args.basedir, 'gen'),
                                                        max_to_keep=5)


    mask_value = vocabs.get("[MASK]", vocabs.get("<MASK>", -1))
    if mask_value == -1:
        logger.error("We could not find a suitable masking token in the vocab")
        return

    if args.restart:
        # The global step gets automatically updated here
        # so we dont have to worry about our LR regimen
        gen_checkpoint.restore(gen_checkpoint_manager.latest_checkpoint)
        discrim_checkpoint.restore(discrim_checkpoint_manager.latest_checkpoint)

    def _replicated_train_step(inputs):
        """This runs on a single replica"""
        noised_x, labels = inputs
        with tf.GradientTape() as tape:
            gen_loss_step, discrim_loss_step, acc = gen_vs_discrim(noised_x, labels, gen_model, discrim_model, mask_value)
            loss_value = (args.gen_loss_scale * gen_loss_step + discrim_loss_step) / num_replicas

        grads = tape.gradient(loss_value, gen_model.trainable_variables + discrim_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip)
        optimizer.apply_gradients(zip(grads, gen_model.trainable_variables + discrim_model.trainable_variables))

        return loss_value, gen_loss_step, discrim_loss_step, acc

    @tf.function
    def _distributed_train_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        loss, gen_loss, discrim_loss, acc = strategy.run(_replicated_train_step, args=(inputs,))
        sum_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        sum_gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
        sum_discrim_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, discrim_loss, axis=None)
        sum_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, acc, axis=None)
        return sum_loss, sum_gen_loss, sum_discrim_loss, sum_acc

    def _replicated_test_step(inputs):
        """This runs on a single replica"""
        noised_x, labels = inputs
        gen_loss_step, discrim_loss_step, acc = gen_vs_discrim(noised_x, labels, gen_model, discrim_model, mask_value)
        loss_value = (args.gen_loss_scale * gen_loss_step + discrim_loss_step) / num_replicas
        return loss_value, gen_loss_step, discrim_loss_step, acc

    @tf.function
    def _distributed_test_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        loss, gen_loss, discrim_loss, acc = strategy.run(_replicated_test_step, args=(inputs,))
        sum_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        sum_gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
        sum_discrim_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, discrim_loss, axis=None)
        sum_acc = strategy.reduce(tf.distribute.ReduceOp.SUM, acc, axis=None)
        return sum_loss, sum_gen_loss, sum_discrim_loss, sum_acc
    # This is the training loop
    start_epoch = 0
    timer = Timer()
    with strategy.scope():

        for epoch in range(start_epoch, args.epochs):
            SET_TRAIN_FLAG(True)
            logger.info('Starting epoch %d', epoch + 1)
            avg_loss = Average('average_train_loss')
            avg_gen_loss = Average('average_gen_loss')
            avg_discrim_loss = Average('average_discrim_loss')
            avg_acc = Average('average_train_acc')

            metrics = {}
            timer.start()
            train_iter = iter(train_loader)
            for i in range(steps_per_epoch):
                loss, gen_loss, discrim_loss, acc = _distributed_train_step(next(train_iter))
                avg_loss.update(loss.numpy().item())
                avg_gen_loss.update(gen_loss.numpy().item())
                avg_discrim_loss.update(discrim_loss.numpy().item())
                avg_acc.update(acc.numpy().item())

                tf.summary.scalar("train_loss", data=loss, step=optimizer.iterations)
                tf.summary.scalar("train_gen_loss", data=gen_loss, step=optimizer.iterations)
                tf.summary.scalar("train_discrim_loss", data=discrim_loss, step=optimizer.iterations)
                tf.summary.scalar("train_acc", data=acc, step=optimizer.iterations)

                if args.convert_only:
                    logger.warning("Convert only flag specified.  Stopping after one step")
                    steps = optimizer.iterations.numpy()
                    npz_checkpoint = os.path.join(args.basedir, f'discrim-step-{steps}.npz')
                    save_tlm_npz(discrim_model, npz_checkpoint)
                    npz_checkpoint = os.path.join(args.basedir, f'gen-step-{steps}.npz')
                    save_tlm_npz(gen_model, npz_checkpoint)
                    return

                if (i + 1) % report_on == 0:
                    logging.info(avg_loss)
                    logging.info(avg_gen_loss)
                    logging.info(avg_discrim_loss)
                    logging.info(avg_acc)
                if (i + 1) % update_on == 0:
                    elapsed = timer.elapsed(True)
                    logging.info('elapsed time this epoch %d min', elapsed)
                    logging.info('elapsed step time %f steps/min', i/elapsed)
                    gen_checkpoint_manager.save()
                    discrim_checkpoint_manager.save()

                    if args.npz:
                        steps = optimizer.iterations.numpy()
                        npz_checkpoint = os.path.join(args.basedir, f'discrim-step-{steps}.npz')
                        save_tlm_npz(discrim_model, npz_checkpoint)
                        npz_checkpoint = os.path.join(args.basedir, f'gen-step-{steps}.npz')
                        save_tlm_npz(gen_model, npz_checkpoint)

            # This is the average training token-level loss across all machines
            # This is the token-level training perplexity
            metrics['train_elapsed_min'] = timer.elapsed(True)
            metrics['average_train_loss'] = avg_loss.avg
            metrics['average_gen_loss'] = avg_gen_loss.avg
            metrics['average_discrim_loss'] = avg_discrim_loss.avg
            metrics['average_train_acc'] = avg_acc.avg
            metrics['lr'] = float(lr_sched(tf.cast(optimizer.global_step, tf.float32)).numpy().item())

            avg_valid_loss = Average('average_valid_loss')
            avg_valid_gen_loss = Average('average_valid_gen_loss')
            avg_valid_discrim_loss = Average('average_valid_discrim_loss')
            avg_valid_acc = Average('average_valid_acc')

            timer.start()
            SET_TRAIN_FLAG(False)
            valid_iter = iter(valid_loader)
            for i in range(steps_per_valid_epoch):
                valid_loss, valid_gen_loss, valid_discrim_loss, valid_acc = _distributed_test_step(next(valid_iter))
                tf.summary.scalar('valid_loss', data=valid_loss, step=optimizer.iterations)
                tf.summary.scalar('valid_gen_loss', data=valid_gen_loss, step=optimizer.iterations)
                tf.summary.scalar('valid_discrim_loss', data=valid_discrim_loss, step=optimizer.iterations)
                tf.summary.scalar('valid_acc', data=valid_acc, step=optimizer.iterations)
                avg_valid_loss.update(valid_loss.numpy().item())
                avg_valid_gen_loss.update(valid_gen_loss.numpy().item())
                avg_valid_discrim_loss.update(valid_discrim_loss.numpy().item())
                avg_valid_acc.update(valid_acc.numpy().item())

            metrics['valid_elapsed_min'] = timer.elapsed(True)
            metrics['average_valid_loss'] = avg_valid_loss.avg
            metrics['average_valid_gen_loss'] = avg_valid_gen_loss.avg
            metrics['average_valid_discrim_loss'] = avg_valid_discrim_loss.avg
            metrics['average_valid_acc'] = avg_valid_acc.avg
            logger.info(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train()

