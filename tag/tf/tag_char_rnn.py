import tensorflow as tf
import numpy as np
import math
import w2v
from data import conllBuildVocab
from data import conllSentsToIndices
from data import batch
from data import revlut
from data import validSplit
from model import TaggerModel, vizEmbeddings
from train import Trainer
import time

DEF_BATCHSZ = 50
DEF_TSF = './data/twpos-data-v0.3/oct27.splits/oct27.train'
DEF_VSF = './data/twpos-data-v0.3/oct27.splits/oct27.dev'
DEF_ESF = './data/twpos-data-v0.3/oct27.splits/oct27.test'
DEF_FILE_OUT = 'rnn-tagger.model'
DEF_EVAL_OUT = 'rnn-tagger-test.txt'
DEF_PATIENCE = 10
DEF_RNN = 'blstm'
DEF_NUM_RNN = 1
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.32
DEF_CFILTSZ = '1,2,3,4,5,7'
DEF_HSZ = 100
DEF_CHARSZ = 16
DEF_WSZ = 50
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0
DEF_UNIF = 0.25
DEF_PDROP = 0.5
DEF_MXLEN = 40
DEF_VALSPLIT = 0.15
DEF_EMBED = None
DEF_CEMBED = None

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', DEF_ETA, 'Initial learning rate.')
flags.DEFINE_string('embed', DEF_EMBED, 'Word2Vec embeddings file')
flags.DEFINE_string('cembed', DEF_CEMBED, 'Word2Vec char embeddings file')
flags.DEFINE_string('optim', DEF_OPTIM, 'Optim method')
flags.DEFINE_float('dropout', DEF_PDROP, 'Dropout probability')
flags.DEFINE_string('train', DEF_TSF, 'Training file')
flags.DEFINE_string('valid', DEF_VSF, 'Validation file')
flags.DEFINE_string('test', DEF_ESF, 'Test file')
flags.DEFINE_string('rnn', DEF_RNN, 'RNN type')
flags.DEFINE_integer('numrnn', DEF_NUM_RNN, 'The depth of stacked RNNs')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_float('unif', DEF_UNIF, 'Initializer bounds for embeddings')
flags.DEFINE_float('clip', DEF_CLIP, 'Gradient clipping cutoff')
flags.DEFINE_integer('epochs', DEF_EPOCHS, 'Number of epochs')
flags.DEFINE_integer('batchsz', DEF_BATCHSZ, 'Batch size')
flags.DEFINE_integer('mxlen', DEF_MXLEN, 'Max length')
flags.DEFINE_string('cfiltsz', DEF_CFILTSZ, 'Character filter sizes')
#flags.DEFINE_integer('charsz', 150, 'Char embedding depth')
flags.DEFINE_integer('charsz', DEF_CHARSZ, 'Char embedding depth')
flags.DEFINE_integer('patience', DEF_PATIENCE, 'Patience')
flags.DEFINE_integer('hsz', DEF_HSZ, 'Hidden layer size')
flags.DEFINE_integer('wsz', DEF_WSZ, 'Word embedding depth')
flags.DEFINE_float('valsplit', DEF_VALSPLIT, 'Validation split if no valid set')
#flags.DEFINE_string('cfiltsz', '0', 'Character filter sizes')
flags.DEFINE_boolean('cbow', False, 'Do CBOW for characters')
flags.DEFINE_string('save', DEF_FILE_OUT, 'Save basename')
#flags.DEFINE_boolean('crf', False, 'Use CRF on top')



maxw, vocab_ch, vocab_word = conllBuildVocab([FLAGS.train, 
                                              FLAGS.test, 
                                              FLAGS.valid])

maxw = min(maxw, FLAGS.mxlen)
print('Max word length %d' % maxw)


if FLAGS.cbow is True:
    print('Using CBOW char embeddings')
    FLAGS.cfiltsz = '0'
else:
    print('Using convolutional char embeddings')
word_vec = None
if FLAGS.embed:
    word_vec = w2v.Word2VecModel(FLAGS.embed, vocab_word, FLAGS.unif)

if FLAGS.cembed:
    print('Using pre-trained character embeddings ' + FLAGS.cembed)
    char_vec = w2v.Word2VecModel(FLAGS.cembed, vocab_ch, FLAGS.unif)
    FLAGS.charsz = char_vec.dsz
    if FLAGS.charsz != FLAGS.wsz and FLAGS.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, and have provided pre-trained char vector embeddings.  To make this work, setting word vector size to character vector size ' + FLAGS.charsz)
        opt.wsz = opt.charsz
else:
    if FLAGS.charsz != FLAGS.wsz and FLAGS.cbow is True:
        print('Warning, you have opted for CBOW char embeddings, but have provided differing sizes for char embedding depth and word depth.  This is not possible, forcing char embedding depth to be word depth ' + FLAGS.wsz)
        FLAGS.charsz = FLAGS.wsz

    char_vec = w2v.RandomInitVecModel(FLAGS.charsz, vocab_ch, FLAGS.unif)


f2i = {"<PAD>":0}

opts = { 'batchsz': FLAGS.batchsz,
         'mxlen': FLAGS.mxlen }

ts, f2i, _ = conllSentsToIndices(FLAGS.train, word_vec, char_vec, maxw, f2i, opts)
print('Loaded  training data')

if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i,_ = conllSentsToIndices(FLAGS.valid, word_vec, char_vec, maxw, f2i, opts)
else:
    ts, vs = validSplit(ts, FLAGS.valsplit)
    print('Created validation split')


es, f2i,txts = conllSentsToIndices(FLAGS.test, word_vec, char_vec, maxw, f2i, opts)
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
                     FLAGS.mxlen,
                     maxw,
                     FLAGS.rnn,
                     FLAGS.wsz,
                     FLAGS.hsz,
                     FLAGS.cfiltsz)


        trainer = Trainer(sess, model, FLAGS.outdir, FLAGS.optim, FLAGS.eta)

        train_writer = trainer.writer()

        vizEmbeddings(char_vec, word_vec, FLAGS.outdir, train_writer)

        init = tf.global_variables_initializer()
        sess.run(init)
        trainer.prepare(tf.train.Saver())

        max_acc = 0
        last_improved = 0
        for i in range(FLAGS.epochs):
            print('Training epoch %d' % (i+1))
            trainer.train(ts, FLAGS.dropout, FLAGS.batchsz)
            avg_loss, this_acc = trainer.test(vs, FLAGS.batchsz, 'Validation')
            if this_acc > max_acc:
                max_acc = this_acc
                last_improved = i
                print('Highest dev acc achieved yet -- writing model')
                trainer.checkpoint(FLAGS.save)
            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest dev acc %.2f' % (max_acc * 100.))
        print('=====================================================')
        print('Evaluating best model on test data')
        print('=====================================================')
        trainer.recover_last_checkpoint()
        #best_model = tf.train.latest_checkpoint(FLAGS.outdir + "/train/")
        #print("Reloading " + best_model)
        #saver.restore(sess, best_model)
        avg_loss, this_acc = trainer.test(es, 1, 'Test')
        print("-----------------------------------------------------")
        print('Test loss %.2f' % (avg_loss))
        print('Test acc %.2f' % (this_acc * 100.))
        print('=====================================================')
        # Write out model, graph and saver for future inference
        model.save(sess, FLAGS.outdir, FLAGS.save)
