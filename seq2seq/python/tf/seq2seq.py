import tensorflow as tf
import numpy as np
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from w2v import Word2VecModel
from data import load_sentences, build_vocab
from utils import *
from model import *
from train import Trainer
import time
MAX_EXAMPLES = 5
SAMPLE_PRUNE_INIT = 5
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.001, 'Initial learning rate.')
flags.DEFINE_string('embed1', None, 'Word2Vec embeddings file (1)')
flags.DEFINE_string('embed2', None, 'Word2Vec embeddings file (2)')
flags.DEFINE_string('rnntype', 'lstm', '(lstm|gru)')
flags.DEFINE_string('optim', 'adam', 'Optim method')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_string('train', None, 'Training file')
flags.DEFINE_string('valid', None, 'Validation file')
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_float('unif', 0.25, 'Initializer bounds for embeddings')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('nreset', 2, 'Number of resets for patience')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_integer('patience', 10, 'Patience')
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_boolean('clean', True, 'Do cleaning')
flags.DEFINE_float('clip', 1, 'Gradient clipping')
flags.DEFINE_integer('layers', 1, 'Number of LSTM layers for encoder/decoder')
flags.DEFINE_boolean('sharedv', False, 'Share vocab between source and destination')
flags.DEFINE_boolean('showex', True, 'Show generated examples every few epochs')
flags.DEFINE_boolean('sample', False, 'If showing examples, sample?')
flags.DEFINE_boolean('seq2seqlib', False, 'Use seq2seq library underneath')
flags.DEFINE_boolean('attn', False, 'Use attention')

# TODO: Allow best path, not just sample path
def show_batch(model, es, sess, rlut1, rlut2, embed2, sample):
    sz = len(es)
    rnum = int((sz - 1) * np.random.random_sample())
    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']
    
    batch = es[rnum]

    src_array = batch['src']
    tgt_array = batch['tgt']
    
    i = 0
    for src_i,tgt_i in zip(src_array, tgt_array):

        if i > MAX_EXAMPLES:
            break
        i += 1
        print('========================================================================')
        sent = lookup_sentence(rlut1, src_i, True)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = np.zeros((1,FLAGS.mxlen))

        next_value = GO
        src_i = src_i.reshape(1, -1)
        for j in range(FLAGS.mxlen):
            dst_i[0,j] = next_value
            probv = model.step(sess, src_i, dst_i)
            output = probv[j].squeeze()
            if sample is False:
                next_value = np.argmax(output)
            else:
                # This method cuts low probability words out of the dists
                # dynamically.  Ideally, we would also use a beam over several
                # paths and pick the most likely path at the end, but this
                # can be done in a separate program, not necessary to train
                next_value = beam_multinomial(SAMPLE_PRUNE_INIT, output)
            
            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
f2i = {}

v1 = [0]
v2 = [1]

if FLAGS.sharedv is True:
    v1.append(1)
    v2.append(0)

vocab1 = build_vocab(v1, {FLAGS.train, FLAGS.test})
vocab2 = build_vocab(v2, {FLAGS.train, FLAGS.test})

embed1 = Word2VecModel(FLAGS.embed1, vocab1, FLAGS.unif)

print('Loaded word embeddings: ' + FLAGS.embed1)

if FLAGS.embed2 is None:
    print('No embed2 found, using embed1 for both')
    args.embed2 = args.embed1

embed2 = Word2VecModel(FLAGS.embed2, vocab2, FLAGS.unif)
print('Loaded word embeddings: ' + FLAGS.embed2)

ts = load_sentences(FLAGS.train, embed1.vocab, embed2.vocab, FLAGS.mxlen, FLAGS.batchsz)
es = load_sentences(FLAGS.test, embed1.vocab, embed2.vocab, FLAGS.mxlen, FLAGS.batchsz)
rlut1 = revlut(embed1.vocab)
rlut2 = revlut(embed2.vocab)

# For now, if attention is on, use library
if FLAGS.attn:
    print('Forcing library option')
    FLAGS.seq2seqlib = True

# Use library-backed seq2seq routine
seq2seq = Seq2SeqLib() if FLAGS.seq2seqlib else Seq2SeqModel()

#with tf.device('/cpu:0'):
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        seq2seq.params(embed1, embed2, FLAGS.mxlen, FLAGS.hsz, FLAGS.layers, FLAGS.attn, FLAGS.rnntype)

        trainer = Trainer(seq2seq, FLAGS.optim, FLAGS.eta, FLAGS.clip)
        train_writer = tf.summary.FileWriter(FLAGS.outdir + "/train", sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        trainer.prepare(tf.train.Saver())

        err_min = 1
        last_improved = 0
        reset = 0
        #show_batch(seq2seq, es, sess, rlut1, rlut2, embed2, True)

        for i in range(FLAGS.epochs):
            print('Training epoch %d' % (i+1))
            trainer.train(ts, sess, train_writer, FLAGS.dropout)
            if FLAGS.showex:
                show_batch(seq2seq, es, sess, rlut1, rlut2, embed2, FLAGS.sample)
            err_rate = trainer.test(es, sess)

            if err_rate < err_min:
                last_improved = i
                err_min = err_rate
                print('Lowest error achieved yet -- writing model')
                seq2seq.save(sess, FLAGS.outdir, 'seq2seq')

            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break
