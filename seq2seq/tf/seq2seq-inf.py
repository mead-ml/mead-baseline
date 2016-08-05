import tensorflow as tf
import numpy as np
from w2v import Word2VecModel
from data import buildVocab
from data import sentsToIndices
from utils import *
from model import Seq2SeqModel
from train import Trainer
import time

MAX_EXAMPLES = 5
SAMPLE_PRUNE_INIT = 5
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_string('indir', 'in', 'Directory where model resides')
flags.DEFINE_boolean('sample', False, 'If showing examples, sample?')

# TODO: Allow best path, not just sample path
def showBatch(model, es, sess, rlut1, rlut2, vocab, sample):
    sz = len(es)
    rnum = int((sz - 1) * np.random.random_sample())
    GO = vocab['<GO>']
    EOS = vocab['<EOS>']
    
    batch = es[rnum]

    src_array = batch['src']
    tgt_array = batch['tgt']
    
    i = 0
    for src_i,tgt_i in zip(src_array, tgt_array):

        if i > MAX_EXAMPLES:
            break
        i += 1

        print('========================================================================')
        sent = lookupSent(rlut1, src_i, True)
        print('[OP] %s' % sent)
        sent = lookupSent(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = np.zeros((1,FLAGS.mxlen))

        next_value = GO
        src_i = src_i.reshape(1, -1)
        for j in range(FLAGS.mxlen):
            dst_i[0,j] = next_value
            probv = model.step(sess, src_i, dst_i)
            output = probv[j].squeeze()
            # This method cuts low probability words out of the dists
            # dynamically.  Ideally, we would also use a beam over several
            # paths and pick the most likely path at the end, but this
            # can be done in a separate program, not necessary to train
            if sample is False:
                next_value = np.argmax(output)
            else:
                next_value = beamMultinomial(SAMPLE_PRUNE_INIT, output)
            if next_value == EOS:
                break

        sent = lookupSent(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
f2i = {}



opts = { 'batchsz': FLAGS.batchsz,
         'mxlen': FLAGS.mxlen }

seq2seq = Seq2SeqModel()
BASE = 'seq2seq'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        seq2seq.restore(sess, FLAGS.indir, BASE, FLAGS.mxlen)
        rlut1 = revlut(seq2seq.vocab1)
        rlut2 = revlut(seq2seq.vocab2)
        es = sentsToIndices(FLAGS.test, seq2seq.vocab1, seq2seq.vocab2, opts)
        init = tf.initialize_all_variables()
        sess.run(init)
        showBatch(seq2seq, es, sess, rlut1, rlut2, seq2seq.vocab2, FLAGS.sample)
