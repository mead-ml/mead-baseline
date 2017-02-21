import tensorflow as tf
import numpy as np
import math
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import w2v
from data import conll_build_vocab
from data import conll_load_sentences
from data import batch
from data import revlut
from data import valid_split
from model import TaggerModel, viz_embeddings
from train import Trainer
import time

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.001, 'Initial learning rate.')
flags.DEFINE_string('embed', None, 'Word2Vec embeddings file')
flags.DEFINE_string('cembed', None, 'Word2Vec char embeddings file')
flags.DEFINE_string('optim', 'sgd', 'Optim method')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_string('train', '', 'Training file')
flags.DEFINE_string('valid', '', 'Validation file')
flags.DEFINE_string('test', '', 'Test file')
flags.DEFINE_string('rnn', 'blstm', 'RNN type')
flags.DEFINE_integer('numrnn', 1, 'The depth of stacked RNNs')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_string('conll_output', 'rnn-tagger-test.txt', 'Place to put test CONLL file')
flags.DEFINE_float('unif', 0.25, 'Initializer bounds for embeddings')
flags.DEFINE_float('clip', 5, 'Gradient clipping cutoff')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', -1, 'Max sentence length')
flags.DEFINE_integer('mxwlen', 40, 'Max word length')
flags.DEFINE_string('cfiltsz', '1,2,3,4,5,7', 'Character filter sizes')
flags.DEFINE_integer('charsz', 16, 'Char embedding depth')
flags.DEFINE_integer('patience', 70, 'Patience')
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_integer('wsz', 30, 'Word embedding depth (per parallel conv)')
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')
flags.DEFINE_boolean('cbow', False, 'Do CBOW for characters')
flags.DEFINE_string('save', 'rnn-tagger.model', 'Save basename')
flags.DEFINE_boolean('crf', False, 'Use CRF on top')
flags.DEFINE_integer('fscore', 0, 'Use F-score in metrics and early stopping')
flags.DEFINE_boolean('viz', False, 'Set up LUT vocabs for Tensorboard')
flags.DEFINE_integer('test_thresh', 10, 'How many epochs improvement required before testing')
maxs, maxw, vocab_ch, vocab_word = conll_build_vocab([FLAGS.train, 
                                                      FLAGS.test, 
                                                      FLAGS.valid])

maxw = min(maxw, FLAGS.mxwlen)
maxs = min(maxs, FLAGS.mxlen) if FLAGS.mxlen > 0 else maxs
print('Max sentence length %d' % maxs)
print('Max word length %d' % maxw)

# Vocab LUTs
word_vocab = None
char_vocab = None


if FLAGS.cbow is True:
    print('Using CBOW char embeddings')
    FLAGS.cfiltsz = '0'
else:
    print('Using convolutional char embeddings')

word_vec = None
if FLAGS.embed:
    word_vec = w2v.Word2VecModel(FLAGS.embed, vocab_word, FLAGS.unif)
    word_vocab = word_vec.vocab

if FLAGS.cembed:
    print('Using pre-trained character embeddings ' + FLAGS.cembed)
    char_vec = w2v.Word2VecModel(FLAGS.cembed, vocab_ch, FLAGS.unif)
    char_vocab = char_vec.vocab

    FLAGS.charsz = char_vec.dsz
    if FLAGS.charsz != FLAGS.wsz and FLAGS.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, and have provided pre-trained char vector embeddings.  To make this work, setting word vector size to character vector size %d' % FLAGS.charsz)
        FLAGS.wsz = FLAGS.charsz
else:
    if FLAGS.charsz != FLAGS.wsz and FLAGS.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, but have provided differing sizes for char embedding depth and word depth.  This is not possible, forcing char embedding depth to be word depth ' + FLAGS.wsz)
        FLAGS.charsz = FLAGS.wsz

    char_vec = w2v.RandomInitVecModel(FLAGS.charsz, vocab_ch, FLAGS.unif)
    char_vocab = char_vec.vocab

f2i = {"<PAD>":0}



ts, f2i, _ = conll_load_sentences(FLAGS.train, word_vocab, char_vocab, maxs, maxw, f2i)
print('Loaded  training data')

if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i,_ = conll_load_sentences(FLAGS.valid, word_vocab, char_vocab, maxs, maxw, f2i)
else:
    ts, vs = valid_split(ts, FLAGS.valsplit)
    print('Created validation split')


es, f2i,txts = conll_load_sentences(FLAGS.test, word_vocab, char_vocab, maxs, maxw, f2i)
print('Loaded test data')

i2f = revlut(f2i)
print(i2f)
print('Using %d examples for training' % len(ts))
print('Using %d examples for validation' % len(vs))
print('Using %d examples for test' % len(es))


with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():

        model = TaggerModel()
        model.params(f2i,
                     word_vec,
                     char_vec,
                     maxs,
                     maxw,
                     FLAGS.rnn,
                     FLAGS.numrnn,
                     FLAGS.wsz,
                     FLAGS.hsz,
                     FLAGS.cfiltsz,
                     FLAGS.crf)

        trainer = Trainer(sess, model, FLAGS.outdir, FLAGS.optim, FLAGS.eta, i2f, FLAGS.fscore)

        train_writer = trainer.writer()

        if FLAGS.viz:
            viz_embeddings(char_vec, word_vec, FLAGS.outdir, train_writer)

        init = tf.global_variables_initializer()
        sess.run(init)
        model.save_using(tf.train.Saver())
        print('Writing metadata')
        model.save_md(sess, FLAGS.outdir, FLAGS.save)

        saving_metric = 0
        metric_type = "F%d" % FLAGS.fscore if FLAGS.fscore > 0 else "acc"
        last_improved = 0
        for i in range(FLAGS.epochs):
            print('Training epoch %d.' % (i+1))
            if i > 0:
                print('\t(last improvement @ %d)' % (last_improved+1))
            trainer.train(ts, FLAGS.dropout, FLAGS.batchsz)
            this_acc, this_f = trainer.test(vs, FLAGS.batchsz, 'Validation')

            this_metric = this_f if FLAGS.fscore > 0 else this_acc
            if this_metric > saving_metric:
                saving_metric = this_metric
                print('Highest dev %s achieved yet -- writing model' % metric_type)

                if (i - last_improved) > FLAGS.test_thresh:
                    trainer.test(es, 1, 'Test')
                trainer.checkpoint(FLAGS.save)
                last_improved = i
                    
                
            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest dev %s %.2f' % (metric_type, saving_metric * 100.))
        print('=====================================================')
        print('Evaluating best model on test data')
        print('=====================================================')
        trainer.recover_last_checkpoint()
        this_acc, this_f = trainer.test(es, 1, 'Test', FLAGS.conll_output, txts)
        print("-----------------------------------------------------")
        print('Test acc %.2f' % (this_acc * 100.))
        if FLAGS.fscore > 0:
            print('Test F%d %.2f' % (FLAGS.fscore, this_f * 100.))
        print('=====================================================')
        # Write out model, graph and saver for future inference
        model.save_values(sess, FLAGS.outdir, FLAGS.save)
