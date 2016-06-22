import tensorflow as tf
import numpy as np
import w2v
from data import conllBuildVocab
from data import conllSentsToIndices
from data import revlut
from data import validSplit
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
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_string('rnn', 'blstm', 'RNN type')
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')

def tensorToSeq(tensor):
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def seqToTensor(sequence):
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

def createModel(nc, model):

    vsz = model.vsz
    dsz = model.dsz
    # These are going to be (B,T)
    x = tf.placeholder(tf.int32, [None, FLAGS.mxlen], name="x")
    y = tf.placeholder(tf.float32, [None, FLAGS.mxlen, nc], name="y")
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    
    with tf.name_scope("LUT"):
        W = tf.Variable(tf.constant(model.weights, dtype=tf.float32), name = "W")

        e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
        
        with tf.control_dependencies([e0]):
            embed = tf.nn.embedding_lookup(W, x)
            # This is going to be a list of tensors of size (B, T, W)

    with tf.name_scope("Recurrence"):
        # List to tensor, reform as (T, B, W)
        embedseq = tensorToSeq(embed)

    if FLAGS.rnn == 'blstm':
        rnnfwd = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hsz)
        rnnbwd = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hsz)
        
        # Primitive will wrap the fwd and bwd, reverse signal for bwd, unroll
        rnnseq, _, __ = tf.nn.bidirectional_rnn(rnnfwd, rnnbwd, embedseq, dtype=tf.float32)
    else:
        rnnfwd = rnn_cell.BasicLSTMCell(FLAGS.hsz)
        # Primitive will wrap RNN and unroll in time
        rnnseq, _ = tf.nn.rnn(rnnfwd, embedseq, dtype=tf.float32)

    with tf.name_scope("output"):
        # Converts seq to tensor, back to (B,T,W)
        hsz = FLAGS.hsz
        

        if FLAGS.rnn == 'blstm':
            hsz *= 2

        W = tf.Variable(tf.truncated_normal([hsz, nc],
                                            stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

        preds = [tf.nn.softmax(tf.matmul(rnnout, W) + b) for rnnout in rnnseq]
        pred = seqToTensor(preds)
        best = tf.argmax(pred, 2)
    
    return x, y, pkeep, pred, best


def createLoss(pred, best, y):
    gold = tf.argmax(y, 2)
    gold = tf.cast(gold, tf.float32)
    best = tf.cast(best, tf.float32)
    cross_entropy = y * tf.log(pred)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(gold)
    all_total = tf.reduce_sum(mask, name="total")
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    all_loss = tf.reduce_mean(cross_entropy, name="loss")
    err = tf.not_equal(best, gold)
    err = tf.cast(err, tf.float32)
    err *= mask
    all_err = tf.reduce_sum(err)
    return all_loss, all_err, all_total

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

# Fill out the y matrix with 0s for other labels
def fill_y(nc, yidx):
    dense = np.zeros((yidx.shape[0], yidx.shape[1], nc))
    for i in range(yidx.shape[0]):
        for j in range(yidx.shape[1]):
            idx = int(yidx[i, j])
            dense[i, j, idx] = 1
    
    return dense


def train(ts, sess, summary_writer, train_op, global_step, summary_op, loss, err, total):

    total_loss = total_err = total_sum = 0
    seq = np.random.permutation(len(ts))
    start_time = time.time()
        
    for j in seq:
        feed_dict = {x: ts[j]["x"], y: fill_y(len(f2i), ts[j]["y"]), pkeep: (1-FLAGS.dropout)}
        
        _, step, summary_str, lossv, errv, totalv = sess.run([train_op, global_step, summary_op, loss, err, total], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        
        total_err += errv
        total_loss += lossv
        total_sum += totalv

    duration = time.time() - start_time

    print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (float(total_loss)/len(seq), total_err, total_sum, float(total_err)/total_sum, duration))

def test(ts, sess, loss, err, total):

    total_loss = total_err = total_sum = 0
    start_time = time.time()
        
    for j in range(len(ts)):
            
        feed_dict = {x: ts[j]["x"], y: fill_y(len(f2i), ts[j]["y"]), pkeep: 1}
        
        
        lossv, errv, totalv = sess.run([loss, err, total], feed_dict=feed_dict)

        total_loss += lossv
        total_err += errv
        total_sum += totalv
        
    duration = time.time() - start_time

    print('Test (Loss %.4f) (Acc %d/%d = %.2f) (%.3f sec)' % (float(total_loss)/len(ts), total_err, total_sum, float(total_err)/total_sum, duration))
    return float(total_err)/total_sum


vocab = conllBuildVocab([FLAGS.train, FLAGS.test, FLAGS.valid])
w2vModel = w2v.Word2VecModel(FLAGS.embed, vocab, FLAGS.unif)

f2i = {"<PAD>":0}
opts = { 'batchsz': FLAGS.batchsz,
         'mxlen': FLAGS.mxlen }
ts, f2i = conllSentsToIndices(FLAGS.train, w2vModel, f2i, opts)
print('Loaded  training data')

es, f2i = conllSentsToIndices(FLAGS.test, w2vModel, f2i, opts)


if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i = conllSentsToIndices(FLAGS.valid, w2vModel, f2i, opts)
else:
    vs = es

print('Loaded test data')

i2f = revlut(f2i)
print(i2f)

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        x, y, pkeep, model, best = createModel(len(f2i), w2vModel)
        loss, err, total = createLoss(model, best, y)
        loss_summary = tf.scalar_summary("loss", loss)
        
        train_op, global_step = createTrainer(loss)
        
        summary_op = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.outdir + "/train", sess.graph)
        
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        min_err = 1
        last_improved = 0
        for i in range(FLAGS.epochs):

            train(ts, sess, train_writer, train_op, global_step, summary_op, loss, err, total)
            this_err = test(vs, sess, loss, err, total)
            if this_err < min_err:
                min_err = this_err
                last_improved = i
                saver.save(sess, FLAGS.outdir + "/train/rnn-tag-fine.model", global_step=global_step)
            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest dev acc %.2f' % (1-min_err))
        print('=====================================================')
        print('Evaluating best model on test data')
        
        best_model = tf.train.latest_checkpoint(FLAGS.outdir + "/train/")
        print("Reloading " + best_model)
        saver.restore(sess, best_model)
        min_err = test(es, sess, loss, err, total)
              
        print("-----------------------------------------------------")
        print('Highest test acc %.4f' % (1-min_err))
        print('=====================================================')
        print('Evaluating best model on test data')
