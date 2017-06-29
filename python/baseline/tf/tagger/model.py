from baseline.tf.tfy import *
import json
import os
from google.protobuf import text_format
from tensorflow.python.platform import gfile

class TaggerModel:

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(self.char_vocab, f)

    def make_feed_dict(self, x, xch, y=None, do_dropout=False):
        pkeep = 1.0-self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.x: x, self.xch: xch, self.pkeep: pkeep}
        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def restore(self, sess, basename, checkpoint_name=None):
        self.sess = sess
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)
            print('Loaded saver def')

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            print('Imported graph def')

            self.sess.run(saver_def.restore_op_name,
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

    def predict(self, x, xch, lengths):

        feed_dict = self.make_feed_dict(x, xch)
        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:
            probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = self.sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    def params(self, sess, labels, word_vec, char_vec, mxlen, maxw, rnntype, nlayers, wsz, hsz, filtsz, crf, pdrop):

        self.sess = sess
        self.crf = crf
        char_dsz = char_vec.dsz
        nc = len(labels)
        self.x = tf.placeholder(tf.int32, [None, mxlen], name="x")
        self.xch = tf.placeholder(tf.int32, [None, mxlen, maxw], name="xch")
        self.y = tf.placeholder(tf.int32, [None, mxlen], name="y")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")
        self.pdrop_value = pdrop
        self.labels = labels
        self.word_vocab = {}
        if word_vec is not None:
            self.word_vocab = word_vec.vocab
        self.char_vocab = char_vec.vocab

        #filtsz = [int(filt) for filt in filtsz.split(',')]

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


def create_model(labels, word_embeddings, char_embeddings, **kwargs):
    hsz = int(kwargs['hsz'])
    layers = int(kwargs.get('layers', 1))
    rnntype = kwargs.get('rnntype', 'lstm')
    print('RNN [%s]' % rnntype)
    maxs = kwargs.get('maxs', 100)
    maxw = kwargs.get('maxw', 100)
    wsz = kwargs.get('wsz', 30)
    filtsz = kwargs.get('cfiltsz')
    crf = bool(kwargs.get('crf', False))
    dropout = float(kwargs.get('dropout', 0.5))
    sess = kwargs.get('sess', tf.Session())
    tagger = TaggerModel()
    tagger.params(sess, labels, word_embeddings, char_embeddings, maxs, maxw, rnntype, layers, wsz, hsz, filtsz, crf, dropout)
    return tagger