import tensorflow as tf
import numpy as np
import w2v
from data import buildVocab
from data import loadTemporalIndices
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
flags.DEFINE_integer('cmotsz', 100, 'Hidden layer size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_string('filtsz', '3,4,5', 'Filter sizes')
flags.DEFINE_boolean('clean', True, 'Do cleaning')
flags.DEFINE_boolean('chars', False, 'Use characters instead of words')
flags.DEFINE_float('valsplit', 0.15, 'Validation split if no valid set')

def createModel(nc, model):

    vsz = model.vsz
    dsz = model.dsz
    x = tf.placeholder(tf.int32, [None, FLAGS.mxlen], name="x")
    y = tf.placeholder(tf.float32, [None, nc], name="y")
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    filtsz = [int(filt) for filt in FLAGS.filtsz.split(',') ]

    with tf.name_scope("LUT"):
        W = tf.Variable(tf.constant(model.weights, dtype=tf.float32), name = "W")

        e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
        
        with tf.control_dependencies([e0]):
            lut = tf.nn.embedding_lookup(W, x)
            expanded = tf.expand_dims(lut, -1)

    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.name_scope('cmot-%s' % fsz):
            siglen = FLAGS.mxlen - fsz + 1
            W = tf.Variable(tf.truncated_normal([fsz, dsz, 1, FLAGS.cmotsz],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[FLAGS.cmotsz], dtype=tf.float32), name="b")
            conv = tf.nn.conv2d(expanded, 
                                W, strides=[1,1,1,1], 
                                padding="VALID", name="conv")

            activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

            mot = tf.nn.max_pool(activation,
                                 ksize=[1, siglen, 1, 1],
                                 strides=[1,1,1,1],
                                 padding="VALID",
                                 name="pool")
            mots.append(mot)
            
    cmotsz_all = FLAGS.cmotsz * len(mots)
    combine = tf.reshape(tf.concat(3, mots), [-1, cmotsz_all])
    with tf.name_scope("dropout"):
        drop = tf.nn.dropout(combine, pkeep)

    with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([cmotsz_all, nc],
                                            stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")
        lin = tf.matmul(drop, W) + b
        best = tf.argmax(lin, 1, name="best")
    
    return x, y, pkeep, lin, best


def createLoss(model, best, y):

    with tf.name_scope("loss"):
        loss = tf.nn.softmax_cross_entropy_with_logits(model, y)
        all_loss = tf.reduce_sum(loss)


    with tf.name_scope("accuracy"):
        correct = tf.equal(best, tf.argmax(y, 1))
        all_right = tf.reduce_sum(tf.cast(correct, "float"), name="accuracy")

    return all_loss, all_right

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
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc))
    dense[xidx, yidx] = 1
    return dense


def train(ts, sess, summary_writer, train_op, global_step, summary_op, loss, acc):

    total_loss = total_corr = total = 0
    seq = np.random.permutation(len(ts))
    start_time = time.time()
        
    for j in seq:
    
        feed_dict = {x: ts[j]["x"], y: fill_y(len(f2i), ts[j]["y"]), pkeep: (1-FLAGS.dropout)}
        
        _, step, summary_str, lossv, accv = sess.run([train_op, global_step, summary_op, loss, acc], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        
        total_corr += accv
        total_loss += lossv
        total += ts[j]["y"].shape[0]

    duration = time.time() - start_time

    print('Train (Loss %.2f) (Acc %d/%d = %.2f) (%.3f sec)' % (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))

def test(ts, sess, loss, acc):

    total_loss = total_corr = total = 0
    start_time = time.time()
        
    for j in range(len(ts)):
            
        feed_dict = {x: ts[j]["x"], y: fill_y(len(f2i), ts[j]["y"]), pkeep: 1}
        
        
        lossv, accv = sess.run([loss, acc], feed_dict=feed_dict)
        total_corr += accv
        total_loss += lossv
        total += ts[j]["y"].shape[0]
        
    duration = time.time() - start_time

    print('Test (Loss %.2f) (Acc %d/%d = %.2f) (%.3f sec)' % (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
    return float(total_corr)/total


vocab = buildVocab([FLAGS.train, FLAGS.test, FLAGS.valid], FLAGS.clean, FLAGS.chars)

w2vModel = w2v.Word2VecModel(FLAGS.embed, vocab, FLAGS.unif)

f2i = {}
opts = { 'batchsz': FLAGS.batchsz,
         'clean': FLAGS.clean,
         'chars': FLAGS.chars,
         'filtsz': [int(filt) for filt in FLAGS.filtsz.split(',')],
         'mxlen': FLAGS.mxlen }
ts, f2i = loadTemporalIndices(FLAGS.train, w2vModel, f2i, opts)
print('Loaded  training data')

if FLAGS.valid is not None:
    print('Using provided validation data')
    vs, f2i = loadTemporalIndices(FLAGS.valid, w2vModel, f2i, opts)
else:
    ts, vs = validSplit(ts, FLAGS.valsplit)
    print('Created validation split')

es, f2i = loadTemporalIndices(FLAGS.test, w2vModel, f2i, opts)
print('Loaded test data')


"""
Train convolutional sentence model
"""
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        x, y, pkeep, model, best = createModel(len(f2i), w2vModel)
        loss, acc = createLoss(model, best, y)
        loss_summary = tf.scalar_summary("loss", loss)
        acc_summary = tf.scalar_summary("accuracy", acc)
        
        train_op, global_step = createTrainer(loss)
        
        summary_op = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.outdir + "/train", sess.graph)
        
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        max_acc = 0
        last_improved = 0
        for i in range(FLAGS.epochs):

            train(ts, sess, train_writer, train_op, global_step, summary_op, loss, acc)
            this_acc = test(vs, sess, loss, acc)
            if this_acc > max_acc:
                max_acc = this_acc
                last_improved = i
                saver.save(sess, FLAGS.outdir + "/train/cnn-sentence-fine.model", global_step=global_step)
            if (i - last_improved) > FLAGS.patience:
                print('Stopping due to persistent failures to improve')
                break


        print("-----------------------------------------------------")
        print('Highest test acc %.2f' % max_acc)
        print('=====================================================')
        print('Evaluating best model on test data')
        
        best_model = tf.train.latest_checkpoint(FLAGS.outdir + "/train/")
        print("Reloading " + best_model)
        saver.restore(sess, best_model)
        test(es, sess, loss, acc)
