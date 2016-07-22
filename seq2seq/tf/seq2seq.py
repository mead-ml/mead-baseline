import tensorflow as tf
import numpy as np
from w2v import Word2VecModel
from data import buildVocab
from data import revlut
from data import sentsToIndices

import time
MAX_EXAMPLES = 5
SAMPLE_PRUNE_INIT = 5
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.001, 'Initial learning rate.')
flags.DEFINE_string('embed1', None, 'Word2Vec embeddings file (1)')
flags.DEFINE_string('embed2', None, 'Word2Vec embeddings file (2)')
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
flags.DEFINE_float('clip', 5, 'Gradient clipping')
flags.DEFINE_boolean('sharedv', False, 'Share vocab between source and destination')
flags.DEFINE_boolean('showex', True, 'Show generated examples every few epochs')
flags.DEFINE_boolean('sample', True, 'If showing examples, sample?')

def tensorToSeq(tensor):
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def seqToTensor(sequence):
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

def createSeq2SeqModel(embed1, embed2):

    # These are going to be (B,T)
    src = tf.placeholder(tf.int32, [None, FLAGS.mxlen], name="src")
    dst = tf.placeholder(tf.int32, [None, FLAGS.mxlen], name="dst")
    tgt = tf.placeholder(tf.int32, [None, FLAGS.mxlen], name="tgt")
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    with tf.name_scope("LUT"):
        Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name = "W")
        Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name = "W")

        ei0 = tf.scatter_update(Wi, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        eo0 = tf.scatter_update(Wo, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        
        with tf.control_dependencies([ei0]):
            embed_in = tf.nn.embedding_lookup(Wi, src)


        with tf.control_dependencies([eo0]):
            embed_out = tf.nn.embedding_lookup(Wo, dst)

    with tf.name_scope("Recurrence"):
        # List to tensor, reform as (T, B, W)
        embed_in_seq = tensorToSeq(embed_in)
        embed_out_seq = tensorToSeq(embed_out)

        rnn_enc = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hsz)
        rnn_dec = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hsz)
        # Primitive will wrap RNN and unroll in time
        rnn_enc_seq, final_encoder_state = tf.nn.rnn(rnn_enc, embed_in_seq, scope='rnn_enc', dtype=tf.float32)
        # Provides the link between the encoder final state and the decoder
        rnn_dec_seq, _ = tf.nn.rnn(rnn_dec, embed_out_seq, initial_state=final_encoder_state, scope='rnn_dec', dtype=tf.float32)

    with tf.name_scope("output"):
        # Leave as a sequence of (T, B, W)
        hsz = FLAGS.hsz

        W = tf.Variable(tf.truncated_normal([hsz, embed2.vsz],
                                            stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[1, embed2.vsz]), name="b")

        preds = [(tf.matmul(rnn_dec_i, W) + b) for rnn_dec_i in rnn_dec_seq]
        probs = [tf.nn.softmax(pred) for pred in preds]
    
    return src, dst, tgt, pkeep, preds, probs #s, best


def createLoss(guess, targets):

    tsparse = tf.unpack(tf.transpose(targets, perm=[1, 0]))

    with tf.name_scope("Loss"):

        log_perp_list = []
        error_list = []
        totalSz = 0
        # For each t in T
        for guess_i, target_i in zip(guess, tsparse):

            # Mask against (B)
            mask = tf.cast(tf.sign(target_i), tf.float32)
            # guess_i = (B, V)
            best_i = tf.cast(tf.argmax(guess_i, 1), tf.int32)
            err = tf.cast(tf.not_equal(best_i, target_i), tf.float32)
            # Gives back (B, V)
            xe = tf.nn.sparse_softmax_cross_entropy_with_logits(guess_i, target_i)

            log_perp_list.append(xe * mask)
            error_list.append(err * mask)
            totalSz += tf.reduce_sum(mask)

        log_perps = tf.add_n(log_perp_list)
        error_all = tf.add_n(error_list)
        log_perps /= totalSz

        cost = tf.reduce_sum(log_perps)
        all_error = tf.reduce_sum(error_all)

        batchSz = tf.cast(tf.shape(tsparse[0])[0], tf.float32)
        return cost/batchSz, all_error, totalSz

def createTrainer(loss):
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if FLAGS.optim == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(FLAGS.eta, 0.95, 1e-6)
    elif FLAGS.optim == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.eta)
    else:
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.eta)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step

def train(ts, sess, summary_writer, train_op, global_step, summary_op, loss, errs, tot):
    total = 0
    total_loss = 0
    total_err = 0
    seq = np.random.permutation(len(ts))
    start_time = time.time()
        
    for j in seq:
        feed_dict = {src: ts[j]["src"], dst: ts[j]["dst"], tgt: ts[j]["tgt"], pkeep: (1-FLAGS.dropout)}
        
        _, step, lossv, errv, totv = sess.run([train_op, global_step, loss, errs, tot], feed_dict=feed_dict)
        total_loss += lossv
        total_err += errv
        total += totv

    duration = time.time() - start_time

    acc = 1.0 - (total_err/total)

    print('Train (Loss %.4f, Acc %.4f) (%.3f sec)' % (float(total_loss)/len(seq), acc, duration))


def test(ts, sess, loss, errs, tot):

    total_loss = total_err = total = 0
    start_time = time.time()
    for j in range(len(ts)):
            
        feed_dict = {src: ts[j]["src"], dst: ts[j]["dst"], tgt: ts[j]["tgt"], pkeep: 1}
        
        
        lossv, errv, totv = sess.run([loss, errs, tot], feed_dict=feed_dict)
        total_loss += lossv
        total_err += errv
        total += totv
        
    duration = time.time() - start_time

    err = total_err/total

    print('Test (Loss %.4f, Acc %.4f) (%.3f sec)' % (float(total_loss)/len(ts), 1.0 - err, duration))

    return err

def lookupSent(rlut, seq, reverse=False):
    s = seq[::-1] if reverse else seq
    return ' '.join([rlut[idx] if rlut[idx] != '<PADDING>' else '' for idx in s])

# Get a sparse index (dictionary) of top values
# Note: mutates input for efficiency
def topk(k, probs):

    lut = {}
    i = 0

    while i < k:
        idx = np.argmax(probs)
        lut[idx] = probs[idx]
        probs[idx] = 0
        i += 1
    return lut

#  Prune all elements in a large probability distribution below the top K
#  Renormalize the distribution with only top K, and then sample n times out of that
def beamMultinomial(k, probs):
    
    tops = topk(k, probs)
    i = 0
    n = len(tops.keys())
    ary = np.zeros((n))
    idx = []
    for abs_idx,v in tops.iteritems():
        ary[i] = v
        idx.append(abs_idx)
        i += 1

    ary /= np.sum(ary)
    sample_idx = np.argmax(np.random.multinomial(1, ary))
    return idx[sample_idx]

# TODO: Allow best path, not just sample path
def showBatch(es, sess, probs, rlut1, rlut2, embed2, sample):
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
        sent = lookupSent(rlut1, src_i, True)
        print('[OP] %s' % sent)
        sent = lookupSent(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = np.zeros((1,FLAGS.mxlen))

        next_value = GO
        src_i = src_i.reshape(1, -1)
        for j in range(FLAGS.mxlen):
            dst_i[0,j] = next_value
            preds = sess.run(probs, feed_dict={
                src:src_i, dst:dst_i
            })

            output = preds[j].squeeze()
            # This method cuts low probability words out of the distributions
            # dynamically.  Ideally, we would also use a beam over several
            # paths and pick the most likely path at the end, but this
            # can be done in a separate program, not necessary to train
            next_value = beamMultinomial(SAMPLE_PRUNE_INIT, output)
            if next_value == EOS:
                break

        sent = lookupSent(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
f2i = {}

v1 = [0]
v2 = [1]

if FLAGS.sharedv is True:
    v1.append(1)
    v2.append(0)

vocab1 = buildVocab(v1, {FLAGS.train, FLAGS.test})
vocab2 = buildVocab(v2, {FLAGS.train, FLAGS.test})

embed1 = Word2VecModel(FLAGS.embed1, vocab1, FLAGS.unif)

print('Loaded word embeddings: ' + FLAGS.embed1)

embed2 = Word2VecModel(FLAGS.embed2, vocab2, FLAGS.unif)
print('Loaded word embeddings: ' + FLAGS.embed2)

opts = { 'batchsz': FLAGS.batchsz,
         'mxlen': FLAGS.mxlen }
ts = sentsToIndices(FLAGS.train, embed1, embed2, opts)
es = sentsToIndices(FLAGS.test, embed1, embed2, opts)
rlut1 = revlut(embed1.vocab)
rlut2 = revlut(embed2.vocab)



with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        src, dst, tgt, pkeep, model, probs = createSeq2SeqModel(embed1, embed2)
        loss, errs, tot = createLoss(model, tgt)
        
        train_op, global_step = createTrainer(loss)
        
        summary_op = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.outdir + "/train", sess.graph)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        err_min = 0
        last_improved = 0
        reset = 0
#        showBatch(es, sess, probs, rlut1, rlut2, embed2, True)

        for i in range(FLAGS.epochs):

            train(ts, sess, train_writer, train_op, global_step, summary_op, loss, errs, tot)
            if FLAGS.showex:
                showBatch(es, sess, probs, rlut1, rlut2, embed2, FLAGS.sample)

            err_rate = test(es, sess, loss, errs, tot)

            if err_rate < err_min:
                last_improved = i
                err_min = err_rate
                print('Lowest error achieved yet -- writing model')
                saver.save(sess, FLAGS.outdir + "/train/seq2seq.model", global_step=global_step)

            if (i - last_improved) > FLAGS.patience:

                if reset < FLAGS.nreset:
                    reset += 1
                    FLAGS.eta *= 0.5
                    last_improved = i
                    print('Patience exhausted, trying again with eta=' + FLAGS.eta)
                    train_op, global_step = createTrainer(loss)
                else:
                    print('Stopping due to persistent failures to improve')
                    break


