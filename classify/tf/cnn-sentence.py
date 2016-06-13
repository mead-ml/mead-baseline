import tensorflow as tf
import numpy as np
import w2v
from data import buildVocab
from data import loadTemporalEmb
from data import revlut
import time
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta0', 0.001, 'Initial learning rate.')
flags.DEFINE_float('unif', 0.25, 'Initializer bounds for embeddings')
flags.DEFINE_integer('epochs', 25, 'Number of epochs')
flags.DEFINE_integer('hsz', 100, 'Hidden layer size')
flags.DEFINE_integer('batchsz', 50, 'Batch size')
flags.DEFINE_string('outdir', 'out', 'Directory to put the output')
flags.DEFINE_integer('mxlen', 100, 'Max length')
flags.DEFINE_string('filtsz', '3,4,5', 'Filter sizes')
flags.DEFINE_boolean('clean', True, 'Do cleaning')
flags.DEFINE_float('pdrop', 0.5, 'Dropout')
flags.DEFINE_integer('patience', 25, 'Patience')
flags.DEFINE_string('embed', None, 'Word2Vec embeddings file')
flags.DEFINE_string('train', None, 'Training file')
flags.DEFINE_string('test', None, 'Test file')
flags.DEFINE_string('optim', 'adam', 'Optim method')

def createModel(nc, model):

    vsz = model.vsz
    dsz = model.dsz
    x = tf.placeholder(tf.float32, [None, FLAGS.mxlen, dsz], name="x")
    y = tf.placeholder(tf.float32, [None, nc], name="y")
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    filtsz = [int(filt) for filt in FLAGS.filtsz.split(',') ]

    with tf.name_scope('expand'):
        expanded = tf.expand_dims(x, -1)

    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.name_scope('cmot-%s' % fsz):
            siglen = FLAGS.mxlen - fsz + 1
            W = tf.Variable(tf.truncated_normal([fsz, dsz, 1, FLAGS.hsz],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[FLAGS.hsz], dtype=tf.float32), name="b")
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
            
    hsz_all = FLAGS.hsz * len(mots)
    combine = tf.reshape(tf.concat(3, mots), [-1, hsz_all])
    with tf.name_scope("dropout"):
        drop = tf.nn.dropout(combine, pkeep)

    with tf.name_scope("output"):
#        W = tf.Variable(tf.truncated_normal([hsz_all, nc],
#                                            stddev = 0.1), name="W")
        W = tf.get_variable("W", shape=[hsz_all, nc], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")
        lin = tf.matmul(drop, W) + b
        # prob = tf.nn.log_softmax(lin, name="prob")
        # tf.nn.in_top_k(prob, labels, 1)
        best = tf.argmax(lin, 1, name="best")
    
    return x, y, pkeep, lin, best


def createLoss(model, best, y):

    with tf.name_scope("loss"):
#        loss = tf.nn.weighted_cross_entropy_with_logits(model, y, pos_weight=1)
        loss = tf.nn.softmax_cross_entropy_with_logits(model, y)
        all_loss = tf.reduce_sum(loss)


    with tf.name_scope("accuracy"):
        correct = tf.equal(best, tf.argmax(y, 1))
        all_right = tf.reduce_sum(tf.cast(correct, "float"), name="accuracy")

    return all_loss, all_right

def createTrainer(loss):
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    if FLAGS.optim == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer()
    elif FLAGS.optim == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.eta0)
    else:
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.eta0)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step

vocab = buildVocab([FLAGS.train, FLAGS.test], True)
w2vModel = w2v.Word2VecModel(FLAGS.embed, vocab, FLAGS.unif)

f2i = {}
opts = { 'batchsz': FLAGS.batchsz,
         'clean': FLAGS.clean,
         'filtsz': [int(filt) for filt in FLAGS.filtsz.split(',')],
         'mxlen': FLAGS.mxlen }
ts, f2i = loadTemporalEmb(FLAGS.train, w2vModel, f2i, opts)
es, f2i = loadTemporalEmb(FLAGS.test, w2vModel, f2i, opts)

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
            
        feed_dict = {x: ts[j]["x"], y: fill_y(len(f2i), ts[j]["y"]), pkeep: (1-FLAGS.pdrop)}

        _, step, summary_str, lossv, accv = sess.run([train_op, global_step, summary_op, loss, acc], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

        total_corr += accv
        total_loss += lossv
        total += ts[j]["y"].shape[0]

    duration = time.time() - start_time

    print('Train (Loss %.2f) (Acc %d/%d = %.2f) (%.3f sec)' % (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))

def test(ts, sess, loss, acc):

    total_loss = total_corr = total = 0
    seq = np.random.permutation(len(ts))
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
        print(summary_op)
        saver = tf.train.Saver()
        train_writer = tf.train.SummaryWriter(FLAGS.outdir + "/train", sess.graph)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        max_acc = 0
        for i in range(FLAGS.epochs):

            train(ts, sess, train_writer, train_op, global_step, summary_op, loss, acc)
            this_acc = test(es, sess, loss, acc)
            if this_acc > max_acc:
                max_acc = this_acc
                saver.save(sess, FLAGS.outdir + "/train", global_step=global_step)

print("-------------------------------------")
print('Best test acc %.2f' % max_acc)
