import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.tensorboard.plugins import projector
from utils import *
import json
import math
import os


def tensor2seq(tensor):
    return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))


def seq2tensor(sequence):
    return tf.transpose(tf.stack(sequence), perm=[1, 0, 2])


def lstm_cell_w_dropout(hsz, pkeep):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=0.0, state_is_tuple=True), output_keep_prob=pkeep)


def stacked_lstm(hsz, pkeep, nlayers):
    return tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pkeep) for _ in range(nlayers)], state_is_tuple=True)


def _viz_embedding(proj_conf, emb, outdir, which):
    emb_conf = proj_conf.embeddings.add()
    emb_conf.tensor_name = '%s/W' % which
    emb_conf.metadata_path = outdir + "/train/metadata-%s.tsv" % which
    write_embeddings_tsv(emb, emb_conf.metadata_path)


def viz_embeddings(char_vec, word_vec, outdir, train_writer):
    print('Setting up word embedding visualization')
    proj_conf = projector.ProjectorConfig()
    _viz_embedding(proj_conf, char_vec, outdir, 'CharLUT')
    if word_vec is not None:
        _viz_embedding(proj_conf, word_vec, outdir, 'WordLUT')
    projector.visualize_embeddings(train_writer, proj_conf)


def skip_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu")

        inputs = inputs + proj
    return inputs


def highway_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("highway-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu-proj")

            W_t = tf.get_variable("W_t", [wsz_all, wsz_all])
            b_t = tf.get_variable("B_t", [1, wsz_all], initializer=tf.constant_initializer(-2.0))
            transform = tf.nn.sigmoid(tf.matmul(inputs, W_t) + b_t, "sigmoid-transform")

        inputs = tf.multiply(transform, proj) + tf.multiply(inputs, 1 - transform)
    return inputs


def char_word_conv_embeddings(char_vec, filtsz, char_dsz, wsz):

    expanded = tf.expand_dims(char_vec, -1)
    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz):

            kernel_shape = [fsz, char_dsz, 1, wsz]
            
            # Weight tying
            W = tf.get_variable("W", kernel_shape)
            b = tf.get_variable("b", [wsz], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(expanded, 
                                W, strides=[1,1,1,1], 
                                padding="VALID", name="conv")
                
            activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

            mot = tf.reduce_max(activation, [1], keep_dims=True)
            # Add back in the dropout
            mots.append(mot)
            
    wsz_all = wsz * len(mots)
    combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])

    # joined = highway_conns(combine, wsz_all, 1)
    joined = skip_conns(combine, wsz_all, 1)
    return joined


def shared_char_word(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        # Zeropad the letters out to half the max filter size, to account for
        # wide convolution.  This way we don't have to explicitly pad the
        # data upfront, which means our Y sequences can be assumed not to
        # start with zeros
        mxfiltsz = np.max(filtsz)
        halffiltsz = int(math.floor(mxfiltsz / 2))
        zeropad = tf.pad(xch_i, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz)

class TaggerModel:

    def save_values(self, sess, outdir, base):
        basename = outdir + '/' + base
        self.saver.save(sess, basename)

    def save_md(self, sess, outdir, base):
        
        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(self.char_vocab, f)
        
    def save(self, sess, outdir, base):
        self.save_md(sess, outdir, base)
        self.save_values(sess, outdir, base)

    def restore(self, sess, indir, base, checkpoint_name=None):
        basename = indir + '/' + base
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)
            print('Loaded saver def')

        with gfile.FastGFile(basename + '.graph', 'r') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            print('Imported graph def')

            sess.run(saver_def.restore_op_name,
                     {saver_def.filename_tensor_name: checkpoint_name})
            self.x = tf.get_default_graph().get_tensor_by_name('x:0')
            self.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            self.y = tf.get_default_graph().get_tensor_by_name('y:0')
            self.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            self.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0') # X
            self.probs = tf.get_default_graph().get_tensor_by_name('output/transpose:0') # X
            try:
                self.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
                print('Found transition matrix in graph, setting crf=True')
                self.crf = True
            except:
                print('Failed to get transition matrix, setting crf=False')
                self.A = None
                self.crf = False

        with open(basename + '.labels', 'r') as f:
            self.labels = json.load(f)

        self.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                self.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            self.char_vocab = json.load(f)


        self.saver = tf.train.Saver(saver_def=saver_def)

    def __init__(self):
        pass

    def save_using(self, saver):
        self.saver = saver

    def _compute_word_level_loss(self, mask):

        nc = len(self.labels)
        # Cross entropy loss
        cross_entropy = tf.one_hot(self.y, nc, axis=-1) * tf.log(tf.nn.softmax(self.probs))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def _compute_sentence_level_loss(self, lengths):

        ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, lengths)
        return tf.reduce_mean(-ll)

    def create_loss(self):
        
        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float32)
            mask = tf.sign(gold)

            lengths = tf.reduce_sum(mask, name="lengths",
                                    reduction_indices=1)
            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss(lengths)
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        return all_loss

    def predict(self, sess, batch):
        
        lengths = batch["length"]
        feed_dict = {self.x: batch["x"], self.xch: batch["xch"], self.pkeep: 1.0}

        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:
            probv, tranv = sess.run([self.probs, self.A], feed_dict=feed_dict)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    def ex2dict(self, batch, pkeep):
        return {
            self.x: batch["x"],
            self.xch: batch["xch"],
            self.y: batch["y"],
            self.pkeep: pkeep
        }

    def params(self, labels, word_vec, char_vec, mxlen, maxw, rnntype, nlayers, wsz, hsz, filtsz, crf=False):

        self.crf = crf
        char_dsz = char_vec.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.int32, [None, mxlen], name="x")
        self.xch = tf.placeholder(tf.int32, [None, mxlen, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, mxlen], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.labels = labels
        self.word_vocab = {}
        if word_vec is not None:
            self.word_vocab = word_vec.vocab
        self.char_vocab = char_vec.vocab

        filtsz = [int(filt) for filt in filtsz.split(',')]

        if word_vec is not None:
            with tf.name_scope("WordLUT"):
                Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")

                we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))

                with tf.control_dependencies([we0]):
                    wembed = tf.nn.embedding_lookup(Ww, self.x, name="embeddings")

        Wc = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch")
        ce0 = tf.scatter_update(Wc, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

        with tf.control_dependencies([ce0]):
            xch_seq = tensor2seq(self.xch)
            cembed_seq = []
            for i, xch_i in enumerate(xch_seq):
                cembed_seq.append(shared_char_word(Wc, xch_i, filtsz, char_dsz, wsz, None if i == 0 else True))
            word_char = seq2tensor(cembed_seq)

        # List to tensor, reform as (T, B, W)
        # Join embeddings along the third dimension
        joint = word_char if word_vec is None else tf.concat(values=[wembed, word_char], axis=2)
        joint = tf.nn.dropout(joint, self.pkeep)
        embedseq = tensor2seq(joint)

        if rnntype == 'blstm':
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            rnnbwd = stacked_lstm(hsz, self.pkeep, nlayers)

            # Primitive will wrap the fwd and bwd, reverse signal for bwd, unroll
            rnnseq, _, __ = tf.contrib.rnn.static_bidirectional_rnn(rnnfwd, rnnbwd, embedseq, dtype=tf.float32)
        else:
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            # Primitive will wrap RNN and unroll in time
            rnnseq, _ = tf.contrib.rnn.static_rnn(rnnfwd, embedseq, dtype=tf.float32)

        with tf.variable_scope("output"):
            # Converts seq to tensor, back to (B,T,W)

            if rnntype == 'blstm':
                hsz *= 2

            W = tf.Variable(tf.truncated_normal([hsz, nc],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1,nc]), name="b")

            preds = [tf.matmul(rnnout, W) + b for rnnout in rnnseq]
            self.probs = seq2tensor(preds)
            self.best = tf.argmax(self.probs, 2)
            # going back to sparse representation
