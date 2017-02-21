import tensorflow as tf
import json
import numpy as np
import math

def lstm_cell(hsz):
    return tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=0.0, state_is_tuple=True)

# TODO: Compare res vs highway
def skip_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu")

        inputs = inputs + proj
    return inputs

def char_word_conv_embeddings(char_vec, filtsz, char_dsz, wsz):

    expanded = tf.expand_dims(char_vec, -1)

    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz):

            kernel_shape = [fsz, char_dsz, 1, wsz]

            # Weight tying
            W = tf.get_variable("W", kernel_shape, initializer=tf.random_normal_initializer())
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
    joined = skip_conns(combine, wsz_all, 1)
    return joined

def shared_char_word(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        mxfiltsz = np.max(filtsz)
        halffiltsz = int(math.floor(mxfiltsz / 2))
        zeropad = tf.pad(xch_i, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        return char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz)


def tensor2seq(tensor):
    return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))

def seq2tensor(sequence):
    return tf.transpose(tf.stack(sequence), perm=[1, 0, 2])

class AbstractLanguageModel(object):

    def __init__(self):
        pass
    def save_using(self, saver):
        self.saver = saver

    def _rnnlm(self, hsz, nlayers, batchsz, inputs, vsz):
        def attn_cell(hsz):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(hsz), output_keep_prob=self.pkeep)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell(hsz) for _ in range(nlayers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(batchsz, tf.float32)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self.initial_state, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hsz])

        with tf.name_scope("Output"):

            softmax_w = tf.get_variable(
                "softmax_w", [hsz, vsz], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [vsz], dtype=tf.float32)

            self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b, name="logits")
        self.final_state = state


    def save_values(self, sess, outdir, base):
        basename = outdir + '/' + base
        self.saver.save(sess, basename)

    def save(self, sess, outdir, base):
        self.save_md(sess, outdir, base)
        self.save_values(sess, outdir, base)

    def create_loss(self):
        with tf.variable_scope("Loss"):
            targets = tf.reshape(self.y, [-1])
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([tf.size(targets)], dtype=tf.float32)])
            loss = tf.reduce_sum(loss) / self.batch_info['batchsz']
            return loss

class WordLanguageModel(AbstractLanguageModel):

    def __init__(self):
        AbstractLanguageModel.__init__(self)

    def params(self, batchsz, nbptt, maxw, word_vec, hsz, nlayers):

        self.x = tf.placeholder(tf.int32, [None, nbptt], name="x")
        self.xch = tf.placeholder(tf.int32, [None, nbptt, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, nbptt], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.word_vocab = word_vec.vocab
        self.batch_info = { "batchsz": batchsz, "nbptt": nbptt, "maxw": maxw}
        vsz = word_vec.vsz + 1

        with tf.name_scope("WordLUT"):
            Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")
            we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
            with tf.control_dependencies([we0]):
                wembed = tf.nn.embedding_lookup(Ww, self.x, name="embeddings")

        inputs = tf.nn.dropout(wembed, self.pkeep)
        inputs = tf.unstack(inputs, num=nbptt, axis=1)
        self._rnnlm(hsz, nlayers, batchsz, inputs, vsz)

    def save_md(self, sess, outdir, base):

        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)
        with open(basename + '-batch_dims.json', 'w') as f:
            json.dump(self.batch_info, f)



class CharCompLanguageModel(AbstractLanguageModel):

    def __init__(self):
        AbstractLanguageModel.__init__(self)

    def params(self, batchsz, nbptt, maxw, vsz, char_vec, filtsz, wsz, hsz, nlayers):

        self.x = tf.placeholder(tf.int32, [None, nbptt], name="x")
        self.xch = tf.placeholder(tf.int32, [None, nbptt, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, nbptt], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.char_vocab = char_vec.vocab
        self.batch_info = {"batchsz": batchsz, "nbptt": nbptt, "maxw": maxw}

        filtsz = [int(filt) for filt in filtsz.split(',')]

        char_dsz = char_vec.dsz

        with tf.name_scope("CharLUT"):
            Wc = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="W")

            ce0 = tf.scatter_update(Wc, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

            with tf.control_dependencies([ce0]):
                xch_seq = tensor2seq(self.xch)
                cembed_seq = []
                for i, xch_i in enumerate(xch_seq):
                    cembed_seq.append(shared_char_word(Wc, xch_i, filtsz, char_dsz, wsz, None if i == 0 else True))
                word_char = seq2tensor(cembed_seq)

            # List to tensor, reform as (T, B, W)
            # Join embeddings along the third dimension
            joint = word_char

        inputs = tf.nn.dropout(joint, self.pkeep)
        inputs = tf.unstack(inputs, num=nbptt, axis=1)
        self._rnnlm(hsz, nlayers, batchsz, inputs, vsz)

    def save_md(self, sess, outdir, base):

        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.char_vocab) > 0:
            with open(basename + '-char.vocab', 'w') as f:
                json.dump(self.char_vocab, f)
        with open(basename + '-batch_dims.json', 'w') as f:
            json.dump(self.batch_info, f)


