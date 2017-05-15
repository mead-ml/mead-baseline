import tensorflow as tf
import numpy as np
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import Word2VecModel
from data import *
from utils import *
from model import *
from tfy import print_batch
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_string('indir', 'out', 'Directory where model resides')

f2i = {}

BASE = 'seq2seq'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        seq2seq = Seq2Seq.from_file(sess, FLAGS.indir, BASE, predict=True)
        rlut1 = revlut(seq2seq.vocab1)
        rlut2 = revlut(seq2seq.vocab2)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        latest = tf.train.latest_checkpoint(FLAGS.indir + "/train/")
        saver.restore(sess, latest)

        es = load_sentences(FLAGS.test, seq2seq.vocab1, seq2seq.vocab2, FLAGS.mxlen)
        es_i = batch(es, 0, FLAGS.batchsz)
        feed_dict = seq2seq.ex2dict(es_i, 1.0)
        probs, best = sess.run([seq2seq.preds, seq2seq.best], feed_dict=feed_dict)
        print_batch(best, rlut2)
