import tensorflow as tf
import numpy as np
import w2v
import time
import json

from model import ConvModel
from data import buildVocab
from data import loadTemporalIndices
from data import validSplit
from utils import revlut, fill_y
from train import Trainer
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.001, 'Initial learning rate.')
flags.DEFINE_string('embed', None, 'Word2Vec embeddings file')
flags.DEFINE_string('optim', 'adam', 'Optim method')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_string('train', None, 'Training file')
flags.DEFINE_string('valid', None, 'Validation file')
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_float('unif', 0.25, 'Initializer bounds for embeddings')
flags.DEFINE_integer('epochs', 25, 'Number of epochs')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_integer('patience', 10, 'Patience')
flags.DEFINE_integer('cmotsz', 100, 'Hidden layer size')
flags.DEFINE_integer('hsz', -1, 'Projection layer size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_string('filtsz', '3,4,5', 'Filter sizes')
flags.DEFINE_boolean('clean', True, 'Do cleaning')
flags.DEFINE_boolean('chars', False, 'Use characters instead of words')
flags.DEFINE_boolean('static', False, 'Fix pre-trained embeddings weights')
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')
flags.DEFINE_string('save', 'classify_sentence_tf', 'Save basename')
vocab = buildVocab([FLAGS.train, FLAGS.test, FLAGS.valid], FLAGS.clean, FLAGS.chars)

unif = 0 if FLAGS.static else FLAGS.unif
w2vModel = w2v.Word2VecModel(FLAGS.embed, vocab, unif)

f2i = {}
opts = { 'batchsz': FLAGS.batchsz,
         'clean': FLAGS.clean,
         'chars': FLAGS.chars,
         'mxlen': FLAGS.mxlen }
ts, f2i = loadTemporalIndices(FLAGS.train, w2vModel.vocab, f2i, opts)
print('Loaded  training data')

if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i = loadTemporalIndices(FLAGS.valid, w2vModel.vocab, f2i, opts)
else:
    ts, vs = validSplit(ts, FLAGS.valsplit)
    print('Created validation split')

es, f2i = loadTemporalIndices(FLAGS.test, w2vModel.vocab, f2i, opts)
print('Loaded test data')

"""
Train convolutional sentence model
"""

model = ConvModel()

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model.params(f2i, w2vModel, FLAGS.mxlen, FLAGS.filtsz, FLAGS.cmotsz, FLAGS.hsz, not FLAGS.static)
        
        trainer = Trainer(sess, model, FLAGS.outdir, FLAGS.optim, FLAGS.eta)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        trainer.prepare(tf.train.Saver())

        max_acc = 0
        last_improved = 0

        for i in range(FLAGS.epochs):
            print('Training epoch %d' % (i+1))
            trainer.train(ts, FLAGS.dropout)
            this_acc = trainer.test(vs, 'Validation')
            if this_acc > max_acc:
                max_acc = this_acc
                last_improved = i
                trainer.checkpoint(FLAGS.save)
                print('Highest dev acc achieved yet -- writing model')

            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest dev acc %.2f' % (max_acc * 100.))
        print('=====================================================')
        print('Evaluating best model on test data:')
        print('=====================================================')
        trainer.recover_last_checkpoint()
        this_acc = trainer.test(es)
        print("-----------------------------------------------------")
        print('Test acc %.2f' % (this_acc * 100.))
        print('=====================================================')
        # Write out model, graph and saver for future inference
        model.save(sess, FLAGS.outdir, FLAGS.save)
