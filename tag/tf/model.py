import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, max_pool2d, fully_connected, flatten, xavier_initializer
def tensorToSeq(tensor):
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def seqToTensor(sequence):
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])



def charWordConvEmbeddings(char_vec, maxw, filtsz, char_dsz, wsz):

    expanded = tf.expand_dims(char_vec, -1)

    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz):

            siglen = maxw - fsz + 1
            kernel_shape =  [fsz, char_dsz, 1, wsz]
            
            # Weight tying
            W = tf.get_variable("W", kernel_shape, initializer=tf.random_normal_initializer())
            b = tf.get_variable("b", [wsz], initializer=tf.constant_initializer(0.0))

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
            
    wsz_all = wsz * len(mots)
    combine = tf.reshape(tf.concat(3, mots), [-1, wsz_all])

    # Make a skip connection

#    with tf.name_scope("proj"):
    with tf.variable_scope("proj"):

        W_p = tf.get_variable("W_p", [wsz_all, wsz_all], initializer=tf.random_normal_initializer())
        b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
        proj = tf.nn.relu(tf.matmul(combine, W_p) + b_p, "proj")

    joined = combine + proj
    return joined


def sharedCharWord(Wch, xch_i, maxw, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        cembed = tf.nn.embedding_lookup(Wch, xch_i)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return charWordConvEmbeddings(cembed, maxw, filtsz, char_dsz, wsz)

def createModel(nc, word_vec, char_vec, mxlen, maxw, rnntype, wsz, hsz, filtsz):

    char_dsz = char_vec.dsz

    # These are going to be (B,T)
    x = tf.placeholder(tf.int32, [None, mxlen], name="x")
    xch = tf.placeholder(tf.int32, [None, mxlen, maxw], name="xch")
    y = tf.placeholder(tf.float32, [None, mxlen, nc], name="y")
    pkeep = tf.placeholder(tf.float32, name="pkeep")

    filtsz = [int(filt) for filt in filtsz.split(',') ]

    if word_vec is not None:
        with tf.name_scope("WordLUT"):
            Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name = "W")

            we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
        
            with tf.control_dependencies([we0]):
                wembed = tf.nn.embedding_lookup(Ww, x)

    with tf.name_scope("CharLUT"):
        Wc = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name = "W")

        ce0 = tf.scatter_update(Wc, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))
        
        with tf.control_dependencies([ce0]):
                xch_seq = tensorToSeq(xch)
                cembed_seq = []
                for i, xch_i in enumerate(xch_seq):
                    cembed_seq.append(sharedCharWord(Wc, xch_i, maxw, filtsz, char_dsz, wsz, None if i == 0 else True))
                word_char = seqToTensor(cembed_seq)

        # List to tensor, reform as (T, B, W)
        # Join embeddings along the third dimension
        joint = word_char if word_vec is None else tf.concat(2, [wembed, word_char])

    with tf.name_scope("Recurrence"):
        embedseq = tensorToSeq(joint)

        if rnntype == 'blstm':
            rnnfwd = tf.nn.rnn_cell.BasicLSTMCell(hsz)
            rnnbwd = tf.nn.rnn_cell.BasicLSTMCell(hsz)
        
            # Primitive will wrap the fwd and bwd, reverse signal for bwd, unroll
            rnnseq, _, __ = tf.nn.bidirectional_rnn(rnnfwd, rnnbwd, embedseq, dtype=tf.float32)
        else:
            rnnfwd = tf.nn.rnn_cell.BasicLSTMCell(hsz)
            # Primitive will wrap RNN and unroll in time
            rnnseq, _ = tf.nn.rnn(rnnfwd, embedseq, dtype=tf.float32)

    with tf.name_scope("output"):
        # Converts seq to tensor, back to (B,T,W)
        
        if rnntype == 'blstm':
            hsz *= 2

        W = tf.Variable(tf.truncated_normal([hsz, nc],
                                            stddev = 0.1), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

        preds = [tf.nn.softmax(tf.matmul(rnnout, W) + b) for rnnout in rnnseq]
        pred = seqToTensor(preds)
        best = tf.argmax(pred, 2)
    
    return x, xch, y, pkeep, pred, best


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
