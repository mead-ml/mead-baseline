import baseline as bl
import baseline.tf.embeddings
import baseline.tf.lm
import numpy as np
import os
import codecs
import argparse
import logging
import tensorflow as tf
import glob

SOS = "<GO>"
EOS = "<EOS>"
LOGGER = logging.getLogger('baseline.timing')
NUM_PRE = 8
NUM_PARALLEL_CALLS = 1
SHUF_BUF_SZ = 1000
EMBED_TYPE = 'default'


def get_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return logging.WARNING


def get_tf_logging_level(ll):
    ll = ll.lower()
    if ll == 'debug':
        return tf.logging.DEBUG
    if ll == 'info':
        return logging.INFO
    return tf.logging.WARN

parser = argparse.ArgumentParser(description='Train a Baseline Language Model with TensorFlow Estimator API')
parser.add_argument('--export_dir', help='Directory for TF export (for serving)', default='./models', type=str)
parser.add_argument('--checkpoint_dir', help='Directory for model checkpoints', default='./tmp/lm', type=str)
# Unused currently, just using default
# parser.add_argument('--model_type', help='What type of model to build', type=str, default='default')
parser.add_argument('--hsz', help='How many hidden units', type=int, default=100)
parser.add_argument('--dsz', help='How many embedding dimensions', type=int, default=150)
parser.add_argument('--steps', help='Number of steps to train', type=int, default=1500)
parser.add_argument('--eval_steps', help='Number of steps to eval', type=int, default=500)
parser.add_argument('--checkpoint_steps', help='Number of steps at which to checkpoint', type=int, default=500)
parser.add_argument('--batchsz', help='Batch size', type=int, default=20)
parser.add_argument('--nctx', help='Size of context or number of steps per line', type=int, default=20)
# Unused currently
# parser.add_argument('--filts', help='Parallel convolution filter widths (if default model)',
#                    type=int,
#                    default=[3, 4, 5],
#                    nargs='+')
parser.add_argument('--train', help='Training file', default="./training-monolingual.tokenized.shuffled/news.en-*")
parser.add_argument('--valid',
                    help='Validation file',
                    default='./heldout-monolingual.tokenized.shuffled/news.en.heldout*')
parser.add_argument('--ll', help='Log level', type=str, default='info')
parser.add_argument('--tf_ll', help='TensorFlow Log level', type=str, default='info')
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--optim', help='Optimizer (sgd, adam) (default is adam)', type=str, default='adam')
parser.add_argument('--gpus', help='Number of GPUs to use', default=1, type=int)
parser.add_argument('--vocab_file', help='Location of a vocab file', default='./vocab-2016-09-10.txt')
args = parser.parse_known_args()[0]
train_batchsz = args.batchsz // args.gpus

logging.basicConfig(level=get_logging_level(args.ll))
tf.logging.set_verbosity(get_tf_logging_level(args.tf_ll))


@bl.str_file
def read_vocab(f):
    return {x.strip(): 1 for x in f.readlines()}


class MultiFileLMDataset(object):
    """Minimally modified from https://github.com/rafaljozefowicz/lm
    """
    def __init__(self, vocab, file_pattern):
        """Constructor

        :param vocab: A dictionary mapping words to indices, <PAD>=0, and assumes a <GO> and <EOS> token defined
        :param file_pattern: This is any pattern that can be used to glob for files
        """
        self._vocab = vocab
        self._file_pattern = file_pattern

    def _parse_sentence(self, line):
        return [self._vocab.get(SOS)] + \
               [self._vocab.get(word, 0) for word in line.strip().split()] + \
               [self._vocab.get(EOS)]

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                yield self._parse_sentence(line)

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps):
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.uint8)
        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0
            for i in range(batch_size):
                tokens_filled = 0
                try:
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sentences)
                        num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                        x[i, tokens_filled:tokens_filled+num_tokens] = streams[i][:num_tokens]
                        y[i, tokens_filled:tokens_filled + num_tokens] = streams[i][1:num_tokens+1]
                        w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams[i] = streams[i][num_tokens:]
                        tokens_filled += num_tokens
                except StopIteration:
                    pass
            if not np.any(w):
                return

            yield x, y, w

    def iterate_once(self, batch_size, num_steps):
        """A generator that does a single pass over the data

        :param batch_size: The number of lines in the batch
        :param num_steps: The number of timesteps in each line to process
        :return: The next batch
        """
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps):
        """A generator that does runs over the data infinitely

        :param batch_size: The number of lines in the batch
        :param num_steps: The number of timesteps in each line to process
        :return: The next batch
        """
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                for file_name in file_patterns:
                    yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

# Read the vocabulary file
vocab = read_vocab(args.vocab_file)

embedding_bundle = bl.embeddings.load_embeddings('x',
                                                 dsz=args.dsz,
                                                 known_vocab=vocab,
                                                 embed_type=EMBED_TYPE)
embeddings_x = embedding_bundle['embeddings']
vocab = embedding_bundle['vocab']

idx2word = {v: k for k, v in vocab.items()}


def word_vecs(x):
    print('[')
    for i in range(len(x)):
        print('\t{}'.format('\t'.join([idx2word[xv] for xv in x[i]])))
    print(']')


def train_input_fn():

    d = MultiFileLMDataset(vocab, args.train)

    def gen():
        return d.iterate_forever(train_batchsz, args.nctx)

    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32, tf.int8),
                                             (tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None])))

    dataset = dataset.prefetch(NUM_PRE)
    it = dataset.make_one_shot_iterator()
    x, y, lengths = it.get_next()
    return {'x': x, 'lengths': lengths}, y


def eval_input_fn():

    d = MultiFileLMDataset(vocab, args.valid)

    def gen():
        return d.iterate_once(args.batchsz, args.nctx)

    dataset = tf.data.Dataset.from_generator(gen,
                                             (tf.int32, tf.int32, tf.int8),
                                             (tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None])))

    dataset = dataset.prefetch(NUM_PRE)
    it = dataset.make_one_shot_iterator()
    x, y, lengths = it.get_next()
    return {'x': x, 'lengths': lengths}, y


def model_fn(features, labels, mode, params):

    if mode == tf.estimator.ModeKeys.EVAL:
        bl.tf.SET_TRAIN_FLAG(False)
        model = bl.create_lang_model({'x': embeddings_x},
                                     x=features['x'],
                                     y=labels,
                                     hsz=args.hsz,
                                     sess=None,
                                     tgt_key='x')
        loss = model.create_test_loss()
        (log_perplexity_tensor,
         log_perplexity_update) = tf.metrics.mean(loss)

        perplexity_tensor = tf.exp(log_perplexity_tensor)
        eval_metric_ops = {
            'perplexity': (perplexity_tensor, log_perplexity_update),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits, loss=loss, eval_metric_ops=eval_metric_ops)
    bl.tf.SET_TRAIN_FLAG(True)
    model = bl.create_lang_model({'x': embeddings_x},
                                 x=features['x'],
                                 y=labels,
                                 hsz=args.hsz,
                                 sess=None,
                                 src_keys=['x'],
                                 tgt_key='x')
    loss = model.create_loss()
    colocate = True if args.gpus > 1 else False
    global_step, train_op = bl.tf.optz.optimizer(loss, optim=args.optim, eta=args.lr, colocate_gradients_with_ops=colocate)
    
    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits,
                                      loss=loss,
                                      train_op=train_op)


checkpoint_dir = '{}-{}'.format(args.checkpoint_dir, os.getpid())
if args.gpus > 1:
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    save_checkpoints_steps=args.checkpoint_steps,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=args.gpus))
else:
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    save_checkpoints_steps=args.checkpoint_steps)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params={})
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=args.steps)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=args.eval_steps)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


