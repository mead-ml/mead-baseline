import tensorflow as tf
from os import sys, path, makedirs
from model import WordLanguageModel, CharCompLanguageModel

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data import *
from train import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 1, 'Initial learning rate.')
flags.DEFINE_string('embed', None, 'Word2Vec embeddings file')
flags.DEFINE_string('cembed', None, 'Word2Vec char embeddings file')
flags.DEFINE_string('optim', 'sgd', 'Optim method')
flags.DEFINE_float('dropout', 0.5, 'Dropout probability')
flags.DEFINE_string('train', '', 'Training file')
flags.DEFINE_string('valid', '', 'Validation file')
flags.DEFINE_string('test', '', 'Test file')
flags.DEFINE_string('rnn', 'lstm', 'RNN type')
flags.DEFINE_integer('numrnn', 2, 'The depth of stacked RNNs')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_float('unif', 0.1, 'Initializer bounds for embeddings')
flags.DEFINE_float('clip', 5, 'Gradient clipping cutoff')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs')
flags.DEFINE_integer('batchsz', 20, 'Batch size')
flags.DEFINE_integer('mxwlen', 40, 'Max word length')
flags.DEFINE_integer('nbptt', 35, 'NBPTT steps')
flags.DEFINE_string('cfiltsz', '1,2,3,4,5,7', 'Character filter sizes')
flags.DEFINE_integer('charsz', 16, 'Char embedding depth')
flags.DEFINE_integer('patience', 70, 'Patience')
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_integer('wsz', 30, 'Word embedding depth (per parallel conv)')
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')
flags.DEFINE_string('save', 'wchar_lm.model', 'Save basename')
flags.DEFINE_integer('fscore', 0, 'Use F-score in metrics and early stopping')
flags.DEFINE_boolean('viz', False, 'Set up LUT vocabs for Tensorboard')
flags.DEFINE_integer('test_thresh', 10, 'How many epochs improvement required before testing')
flags.DEFINE_boolean('char', False, 'Use character-level modeling')

flags.DEFINE_float('decay', 0.5, 'Learning rate decay')
if path.exists(FLAGS.outdir) is False:
    print('Creating path: %s' % (FLAGS.outdir))
    makedirs(FLAGS.outdir)

maxw, vocab_ch, vocab_word, num_words = ptb_build_vocab([FLAGS.train, 
                                                         FLAGS.valid, 
                                                         FLAGS.test])

maxw = min(maxw, FLAGS.mxwlen)
print('Max word length %d' % maxw)

# Vocab LUTs
word_vocab = None
char_vocab = None

# No matter what we will create a vocab for words, since that is what we are emitting
word_vec = None
if FLAGS.embed and FLAGS.char is False:
    word_vec = w2v.Word2VecModel(FLAGS.embed, vocab_word, FLAGS.unif)
    word_vocab = word_vec.vocab
# TODO: Fix this to be a boolean for use word vectors
else:
    print('Creating new embedding weights')
    word_vec = w2v.RandomInitVecModel(FLAGS.hsz, vocab_word, FLAGS.unif)
    word_vocab = word_vec.vocab

# No matter what we will create a vocab for letters, for simplicity, even if unused
if FLAGS.cembed:
    print('Using pre-trained character embeddings ' + FLAGS.cembed)
    char_vec = w2v.Word2VecModel(FLAGS.cembed, vocab_ch, FLAGS.unif)
    char_vocab = char_vec.vocab
    FLAGS.charsz = char_vec.dsz
else:
    char_vec = w2v.RandomInitVecModel(FLAGS.charsz, vocab_ch, FLAGS.unif)
    char_vocab = char_vec.vocab

ts = ptb_load_sentences(FLAGS.train, word_vocab, char_vocab, num_words[0], maxw)
print('Loaded training data')

vs = ptb_load_sentences(FLAGS.valid, word_vocab, char_vocab, num_words[1], maxw)
print('Loaded validation data')

es = ptb_load_sentences(FLAGS.test, word_vocab, char_vocab, num_words[2], maxw)
print('Loaded test data')

print('Using %d examples for training' % num_words[0])
print('Using %d examples for validation' % num_words[1])
print('Using %d examples for test' % num_words[2])


best_valid_perplexity = 100000
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-FLAGS.unif, FLAGS.unif)

    lm = None
    if FLAGS.char is True:
        print('Using character-level modeling')
        lm = CharCompLanguageModel()
        lm.params(FLAGS.batchsz, FLAGS.nbptt, maxw, word_vec.vsz + 1, char_vec, FLAGS.cfiltsz, FLAGS.wsz, FLAGS.hsz, FLAGS.numrnn)
    else:
        print('Using word-level modeling')
        lm = WordLanguageModel()
        lm.params(FLAGS.batchsz, FLAGS.nbptt, maxw, word_vec, FLAGS.hsz, FLAGS.numrnn)

    with tf.Session() as sess:

        trainer = Trainer(sess, lm, FLAGS.outdir, FLAGS.optim, FLAGS.eta, FLAGS.clip)

        init = tf.global_variables_initializer()
        sess.run(init)
        lm.save_using(tf.train.Saver())
        print('Writing metadata')
        lm.save_md(sess, FLAGS.outdir, FLAGS.save)

        last_improved = 0
        for i in range(FLAGS.epochs):
            print('Epoch %d' % (i + 1))
            train_perplexity = trainer.train(ts, 1.0 - FLAGS.dropout)
            valid_perplexity = trainer.test(vs, phase='Validation')

            if valid_perplexity < best_valid_perplexity:
                best_valid_perplexity = valid_perplexity
                print('Lowest dev perplexity achieved yet -- writing model')
                trainer.checkpoint(FLAGS.save)
                last_improved = i

            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break

        print("-----------------------------------------------------")
        print('Lowest dev perplexity %.3f' % best_valid_perplexity)
        print('=====================================================')
        print('Evaluating best model on test data')
        print('=====================================================')
        trainer.recover_last_checkpoint()

        trainer.test(es)
        # Write out model, graph and saver for future inference
        lm.save_values(sess, FLAGS.outdir, FLAGS.save)
