import logging
import time
import os
from argparse import ArgumentParser
import math
from typing import Tuple
import baseline
from eight_mile.utils import str2bool, write_json, read_json
import baseline.tf.embeddings
import baseline.embeddings
from baseline.vectorizers import BPEVectorizer1D
from eight_mile.utils import Average, get_num_gpus_multiworker
from eight_mile.tf.layers import create_distribute_strategy, TransformerEncoderStack, EmbeddingsStack, read_yaml_tf, read_json_tf
from eight_mile.optz import *
from eight_mile.tf.optz import *
from baseline.tf.tfy import SET_TRAIN_FLAG
from eight_mile.tf.serialize import save_tlm_output_npz
import tensorflow as tf
import json
logger = logging.getLogger(__file__)


"""Pre-train a Transformer model in TensorFlow

Read in preprocessed contexts generated by `preproc-tlm` and use a Transformer Language Model (TLM)
to train them across multiple workers.

Sample preprocessed data can be found here: https://www.dropbox.com/s/jir4layu6nzjtqy/wt2.tar.gz

"""


class TransformerTagger(tf.keras.Model):
    def __init__(
        self,
        num_labels: int,
        embeddings: tf.keras.layers.Layer,
        num_heads=8, num_layers=8, d_model=512, d_ff=None, rpr_k=None, activation='gelu', scale=True, ffn_pdrop=0.0,
        layer_norm_eps=1.0e-6, rpr_value_on=True, layer_drop=0.0, dropout=0.1,
        name: str = None, **kwargs

    ):
        super().__init__(name=name)
        self.embeddings = EmbeddingsStack(embeddings)
        self.transformer = TransformerEncoderStack(
            num_heads, d_model=d_model,
            pdrop=dropout, scale=scale,
            layers=num_layers, d_ff=d_ff, rpr_k=rpr_k,
            activation=activation,
            ffn_pdrop=ffn_pdrop,
            layer_norm_eps=layer_norm_eps,
            rpr_value_on=rpr_value_on,
            layer_drop=layer_drop)
        self.output_layer = tf.keras.layers.Dense(num_labels)

    def call(self, inputs):
        input_mask = tf.expand_dims(tf.expand_dims(tf.cast(inputs['x'] != 0, tf.int32), 1), 1)
        embed = self.embeddings(inputs)
        transformed = self.transformer((embed, input_mask))
        output = self.output_layer(transformed)
        return output


def loss_function(model, features, labels):
    logits = model(features)
    loss_mask = tf.cast(labels != 0, tf.float32)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = losses * loss_mask
    losses = tf.reduce_sum(losses)
    non_zero = tf.reduce_sum(loss_mask)
    losses /= non_zero
    return losses


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
    if not yml:
        raise Exception(f"Invalid sample file {sample_md}")
    return yml['num_samples']



def train():
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=str)
    parser.add_argument("--train_dir", type=str, required=True, help='Training directory')
    parser.add_argument("--valid_dir", type=str, required=True, help='Validation directory')
    parser.add_argument("--train_md", type=str, help="Training metadata YAML, defaults to `{train_dir}/md.yml`")
    parser.add_argument("--valid_md", type=str, help="Validation metadata YAML, defaults to `{valid_dir}/md.yml`")
    parser.add_argument("--label_file", type=str, help="JSON file mapping labels to integers", default="labels.json")
    parser.add_argument("--dataset_key", default="tlm",
                        help="dataset key for basedir")
    parser.add_argument("--embed_type", type=str, default='default',
                        choices=["default", "positional", "learned-positional"],
                        help="register label of the embeddings")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension (and embedding dsz)")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--distribute", type=str, default="mirror", choices=["mirror", "tpu", "nccl"])
    parser.add_argument("--tpu_ep", type=str, help="The TPU endpoint if using `distribute=tpu`")
    parser.add_argument("--nctx", type=int, default=256, help="Max input length")
    parser.add_argument("--file_type", default='tfrecord', choices=['json', 'tfrecord'], help="Glob pattern for data")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--subword_model_file", type=str, help="The BPE model file", required=True)
    parser.add_argument("--subword_vocab_file", type=str, help="The BPE subword vocab", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--ffn_pdrop", type=float, default=0.0, help="Dropout in the dense stack")
    parser.add_argument("--layer_drop", type=float, default=0.0, help="LayerDrop to apply")
    parser.add_argument("--optim", default="adamw", type=str, help="Optimizer to use (defaults to adamw)")
    parser.add_argument("--lr", type=float, default=4.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=32, help="Num training epochs")
    parser.add_argument("--restart", type=str2bool, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Num warmup steps")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of checkpoints to save per epoch")
    parser.add_argument('--rpr_k',
                        help='Relative attention positional sizes pass 0 if you dont want relative attention',
                        type=int, default=[8], nargs='+')
    parser.add_argument('--rpr_value_on', type=str2bool, default=True,
                        help="In relative attention, whether add positional correction to values in addition to the "
                             "correction to attention matrix")
    parser.add_argument('--windowed_ra', type=str2bool, default=False, help="whether prevent attention beyond rpr_k")
    parser.add_argument("--strategy", help="Training strategy, defaults to `mirror`", choices=["mirror"])
    parser.add_argument("--npz", help="Should we write out NPZ files?", type=str2bool, default=False)
    parser.add_argument("--tb", help="Turn on tensorboard?", type=str2bool, default=False)
    parser.add_argument("--convert_only", help="Should we just convert this file to NPZ and exit?", type=str2bool, default=False)
    args = parser.parse_args()
    SET_TRAIN_FLAG(True)

    if args.convert_only:
        args.restart = True

    if args.basedir is None:
        args.basedir = f'lm-{args.dataset_key}-bpe-{os.getpid()}'
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
    preproc_data = baseline.embeddings.load_embeddings('x', dsz=args.d_model, known_vocab=vocab['x'],
                                                       preserve_vocab_indices=True,
                                                       embed_type=args.embed_type)
    vocabs = preproc_data['vocab']

    train_md = args.train_md if args.train_md else os.path.join(args.train_dir, 'md.yml')
    num_train_samples = get_num_samples(train_md)
    valid_md = args.valid_md if args.valid_md else os.path.join(args.valid_dir, 'md.yml')
    num_valid_samples = get_num_samples(valid_md)
    labels = read_json_tf(args.label_file)
    num_labels = len(labels)

    def dataset_train_fn(input_context):
        global_batchsz = args.batch_size
        base_batchsz = input_context.get_per_replica_batch_size(global_batchsz)
        ds = get_dataset(args.train_dir, args.file_type, args.num_train_workers).batch(base_batchsz)
        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    train_loader = strategy.experimental_distribute_datasets_from_function(dataset_train_fn)

    def dataset_test_fn(input_context):
        global_batchsz = args.batch_size
        base_batchsz = input_context.get_per_replica_batch_size(global_batchsz)
        ds = get_dataset(args.valid_dir, args.file_type, args.num_train_workers, shuffle=False).batch(base_batchsz)

        return ds.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id
        )
    valid_loader = strategy.experimental_distribute_datasets_from_function(dataset_test_fn)

    os.makedirs(args.basedir, exist_ok=True)
    # We want to make sure to save our input vocab into the basedir for reuse later
    write_json(vocabs, os.path.join(args.basedir, 'vocabs.json'))
    embeddings = {'x': preproc_data['embeddings']}
    logger.info("Loaded embeddings")

    logger.info("Loaded datasets")
    logger.info("Using embedding type [%s]", args.embed_type)
    if len(args.rpr_k) == 0 or args.rpr_k[0] < 1:
        args.rpr_k = None
    elif len(args.rpr_k) == 1:
        args.rpr_k = args.rpr_k[0]

    model = TransformerTagger(num_labels, embeddings, **vars(args))

    logger.info("Loaded model and loss")

    steps_per_epoch = num_train_samples // args.batch_size
    steps_per_valid_epoch = num_valid_samples // args.batch_size
    update_on = steps_per_epoch // args.saves_per_epoch
    report_on = max(10, update_on) // 10
    logger.info(f"Steps per epoch: {steps_per_epoch}. Saving checkpoint every {update_on} steps.")

    lr_decay = CosineDecaySchedulerTensorFlow(steps_per_epoch * args.epochs, lr=args.lr)
    linear_warmup = WarmupLinearSchedulerTensorFlow(args.warmup_steps, lr=args.lr)
    lr_sched = CompositeLRSchedulerTensorFlow(linear_warmup, lr_decay)
    optimizer = EagerOptimizer(loss_function, optim=args.optim, lr_function=lr_sched, weight_decay=args.weight_decay, clip=args.clip, lr=args.lr)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer.optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=args.basedir,
                                                    max_to_keep=5)

    start_epoch = 0
    if args.restart:
        # The global step gets automatically updated here
        # so we dont have to worry about our LR regimen
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        current_step = optimizer.global_step
        start_epoch = current_step // steps_per_epoch

    def _replicated_train_step(inputs):
        """This runs on a single replica"""
        x, y = inputs
        per_replica_loss = optimizer.update(model, {'x': x}, y, num_replicas)
        return per_replica_loss

    @tf.function
    def _distributed_train_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        per_replica_loss = strategy.run(_replicated_train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    def _replicated_test_step(inputs):
        """This runs on a single replica"""
        x, y = inputs
        per_replica_loss = loss_function(model, {'x': x}, y) / num_replicas
        return per_replica_loss

    @tf.function
    def _distributed_test_step(inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Runs across multiple replicas and aggregates the results.

        :param inputs:
        :return:
        """
        per_replica_loss = strategy.run(_replicated_test_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    with strategy.scope():

        for epoch in range(start_epoch, args.epochs):
            SET_TRAIN_FLAG(True)
            logger.info('Starting epoch %d', epoch + 1)
            avg_loss = Average('average_train_loss')
            metrics = {}
            start = time.time()
            train_iter = iter(train_loader)
            for i in range(steps_per_epoch):

                try:
                    loss = _distributed_train_step(next(train_iter))
                    avg_loss.update(loss.numpy().item())
                    tf.summary.scalar("train_loss", data=loss, step=optimizer.global_step)
                except Exception as e:
                    logger.error(f"Exception at training step {i+1}/{steps_per_epoch}. Skipping")
                    pass
                if args.convert_only:
                    logger.warning("Convert only flag specified.  Stopping after one step")
                    steps = optimizer.global_step.numpy()
                    npz_checkpoint = os.path.join(args.basedir, f'checkpoint-step-{steps}.npz')
                    save_tlm_output_npz(model, npz_checkpoint)
                    return

                steps = optimizer.global_step.numpy()
                if (steps + 1) % report_on == 0:
                    logger.info(avg_loss)
                if (steps + 1) % update_on == 0:
                    elapsed = (time.time() - start)/60
                    logger.info('elapsed time this epoch %d min', elapsed)
                    logger.info('elapsed step time %f steps/min', i/elapsed)
                    checkpoint_manager.save()
                    if args.npz:

                        npz_checkpoint = os.path.join(args.basedir, f'checkpoint-step-{steps}.npz')
                        save_tlm_output_npz(model, npz_checkpoint)

            # How much time elapsed in minutes
            elapsed = (time.time() - start)/60
            train_token_loss = avg_loss.avg
            # This is the average training token-level loss across all machines
            # This is the token-level training perplexity
            train_token_ppl = math.exp(train_token_loss)
            metrics['train_elapsed_min'] = elapsed
            metrics['average_train_loss'] = train_token_loss
            metrics['train_ppl'] = train_token_ppl
            metrics['lr'] = float(lr_sched(tf.cast(optimizer.global_step, tf.float32)).numpy().item())

            avg_valid_loss = Average('average_valid_loss')
            start = time.time()
            SET_TRAIN_FLAG(False)
            valid_iter = iter(valid_loader)
            for i in range(steps_per_valid_epoch):
                try:
                    valid_loss = _distributed_test_step(next(valid_iter))
                    tf.summary.scalar('valid_loss', data=valid_loss, step=optimizer.global_step)
                    avg_valid_loss.update(valid_loss.numpy().item())
                except Exception as e:
                    logger.error(f"Exception at validation step {i+1}/{steps_per_valid_epoch}. Skipping")
                    pass

            valid_token_loss = avg_valid_loss.avg
            valid_token_ppl = math.exp(valid_token_loss)

            elapsed = (time.time() - start)/60

            metrics['valid_elapsed_min'] = elapsed
            metrics['average_valid_loss'] = valid_token_loss
            metrics['average_valid_word_ppl'] = valid_token_ppl
            logger.info(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    train()

