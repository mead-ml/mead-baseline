import tensorflow as tf
import numpy as np
import w2v
from data import loadTemporalEmb
from model import ConvModelStatic
from data import buildVocab
from data import validSplit
from utils import revlut
from train import Trainer
import time
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
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')

w2vModel = w2v.Word2VecModel(FLAGS.embed)

f2i = {}
opts = { 'batchsz': FLAGS.batchsz,
         'clean': FLAGS.clean,
         'filtsz': [int(filt) for filt in FLAGS.filtsz.split(',')],
         'mxlen': FLAGS.mxlen }
ts, f2i = loadTemporalEmb(FLAGS.train, w2vModel, f2i, opts)
es, f2i = loadTemporalEmb(FLAGS.test, w2vModel, f2i, opts)
print('Loaded  training data')

if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i = loadTemporalEmb(FLAGS.valid, w2vModel, f2i, opts)
else:
    ts, vs = validSplit(ts, FLAGS.valsplit)
    print('Created validation split')

es, f2i = loadTemporalEmb(FLAGS.test, w2vModel, f2i, opts)
print('Loaded test data')



"""
Train convolutional sentence model
"""

model = ConvModelStatic()

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model.params(f2i, w2vModel, FLAGS.mxlen, FLAGS.filtsz, FLAGS.cmotsz, FLAGS.hsz)
   
        trainer = Trainer(model, FLAGS.optim, FLAGS.eta)
        train_writer = tf.train.SummaryWriter(FLAGS.outdir + "/train", sess.graph)
        init = tf.initialize_all_variables()
        sess.run(init)
        trainer.prepare(tf.train.Saver())

        max_acc = 0
        last_improved = 0

        for i in range(FLAGS.epochs):
            print('Training epoch %d' % (i+1))
            trainer.train(ts, sess, train_writer, FLAGS.dropout)
            this_acc = trainer.test(vs, sess)
            if this_acc > max_acc:
                max_acc = this_acc
                last_improved = i
                trainer.checkpoint(sess, FLAGS.outdir, "cnn-sentence")

            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest validation acc %.4f' % max_acc)
        print('=====================================================')
        print('Evaluating best model on test data:')
        print('=====================================================')
        trainer.recover_last_checkpoint(sess, FLAGS.outdir)
        trainer.test(es, sess)

        # Write out model, graph and saver for future inference
        model.save(sess, FLAGS.outdir, 'cnn-sentence')
