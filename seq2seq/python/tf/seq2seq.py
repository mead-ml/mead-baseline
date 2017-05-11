import tensorflow as tf
import numpy as np
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import Word2VecModel
from data import load_sentences, build_vocab
from utils import *
from tfy import *
from model import *
from train import Trainer
import time


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.001, 'Initial learning rate.')
flags.DEFINE_string('embed1', None, 'Word2Vec embeddings file (1)')
flags.DEFINE_string('embed2', None, 'Word2Vec embeddings file (2)')
flags.DEFINE_string('rnntype', 'lstm', '(lstm|gru)')
flags.DEFINE_string('optim', 'adam', 'Optim method')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_string('train', None, 'Training file')
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_float('unif', 0.25, 'Initializer bounds for embeddings')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_integer('patience', 10, 'Patience')
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_float('clip', 1, 'Gradient clipping')
flags.DEFINE_integer('layers', 1, 'Number of LSTM layers for encoder/decoder')
flags.DEFINE_boolean('sharedv', False, 'Share vocab between source and destination')
flags.DEFINE_boolean('showex', True, 'Show generated examples every few epochs')
flags.DEFINE_boolean('sample', False, 'If showing examples, sample?')
flags.DEFINE_boolean('attn', False, 'Use attention')

f2i = {}

v1 = [0]
v2 = [1]

if FLAGS.sharedv is True:
    v1.append(1)
    v2.append(0)

vocab1 = build_vocab(v1, [FLAGS.train, FLAGS.test])
vocab2 = build_vocab(v2, [FLAGS.train, FLAGS.test])

embed1 = Word2VecModel(FLAGS.embed1, vocab1, FLAGS.unif)

print('Loaded word embeddings: ' + FLAGS.embed1)

if FLAGS.embed2 is None:
    print('No embed2 found, using embed1 for both')
    args.embed2 = args.embed1

embed2 = Word2VecModel(FLAGS.embed2, vocab2, FLAGS.unif)
print('Loaded word embeddings: ' + FLAGS.embed2)

ts = load_sentences(FLAGS.train, embed1.vocab, embed2.vocab, FLAGS.mxlen)
es = load_sentences(FLAGS.test, embed1.vocab, embed2.vocab, FLAGS.mxlen)
rlut1 = revlut(embed1.vocab)
rlut2 = revlut(embed2.vocab)

#with tf.device('/cpu:0'):
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():



        if FLAGS.attn is True:
            seq2seq_creator_fn = Seq2Seq.create_lstm_attn if FLAGS.rnntype.lower() == 'lstm' else Seq2Seq.create_gru_attn
        else:
            seq2seq_creator_fn = Seq2Seq.create_lstm if FLAGS.rnntype.lower() == 'lstm' else Seq2Seq.create_gru

        seq2seq = seq2seq_creator_fn(embed1, embed2, FLAGS.mxlen, FLAGS.hsz, FLAGS.layers)

        trainer = Trainer(seq2seq, FLAGS.optim, FLAGS.eta, FLAGS.clip)
        train_writer = tf.summary.FileWriter(FLAGS.outdir + "/train", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        trainer.prepare(saver)

        err_min = 100
        last_improved = 0

        for i in range(FLAGS.epochs):
            print('Training epoch %d' % (i+1))
            trainer.train(ts, sess, train_writer, FLAGS.dropout, FLAGS.batchsz)
            #print_batch(trainer.best_in_batch(es, sess, FLAGS.batchsz), rlut2)

            err_rate = trainer.test(es, sess, FLAGS.batchsz)

            if err_rate < err_min:
                last_improved = i
                err_min = err_rate
                print('Lowest error achieved yet -- writing model')
                seq2seq.save(sess, FLAGS.outdir, 'seq2seq')
                trainer.checkpoint(sess, FLAGS.outdir, 'seq2seq')

            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break
