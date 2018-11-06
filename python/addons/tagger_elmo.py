from baseline.tf.tfy import *
import json
import os
import sys
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.model import TaggerModel, create_tagger_model, load_tagger_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from baseline.utils import Offsets


class RNNTaggerModelELMoModel(TaggerModel):

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])
        state = {"mxlen": self.mxlen, "maxw": self.maxw, "crf": self.crf, "proj": self.proj, "crf_mask": self.crf_mask, 'span_type': self.span_type}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

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

    def make_input(self, batch_dict, train=False):
        x = batch_dict['x']
        x_lc = batch_dict['x_lc']
        #np.set_printoptions(threshold=20)
        y = batch_dict.get('y', None)
        xch = batch_dict['xch']
        lengths = batch_dict['lengths']

        pkeep = 1.0-self.pdrop_value if train else 1.0
        feed_dict = {self.x: x, self.x_lc: x_lc, self.xch: xch, self.lengths: lengths, self.pkeep: pkeep}
        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    @staticmethod
    def load(basename, **kwargs):
        model = RNNTaggerModelELMoModel()
        model.sess = kwargs.get('sess', tf.Session())
        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.state') as f:
            state = json.load(f)
            model.mxlen = state.get('mxlen', 100)
            model.maxw = state.get('maxw', 100)
            model.crf = bool(state.get('crf', False))
            model.crf_mask = bool(state.get('crf_mask', False))
            model.span_type = state.get('span_type')
            model.proj = bool(state.get('proj', False))

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            model.sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            model.sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.x_lc = tf.get_default_graph().get_tensor_by_name('x_lc:0')
            model.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0')
            model.probs = tf.get_default_graph().get_tensor_by_name('output/Reshape_1:0')  # TODO: rename 
            model.sess.run(tf.get_default_graph().get_operation_by_name('index2word/table_init'))
            try:
                model.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
                #print('Found transition matrix in graph, setting crf=True')
                if not model.crf:
                    print('Warning: meta-data says no CRF but model contains transition matrix!')
                    model.crf = True
            except:
                if model.crf is True:
                    print('Warning: meta-data says there is a CRF but not transition matrix found!')
                model.A = None
                model.crf = False

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        model.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                model.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            model.char_vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)

        return model

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

    def _compute_sentence_level_loss(self):

        if self.crf_mask:
            assert self.span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            A = tf.get_variable(
                "transitions_raw",
                shape=(len(self.labels), len(self.labels)),
                dtype=tf.float32,
                trainable=True
            )

            self.mask = crf_mask(self.labels, self.span_type, Offsets.GO, Offsets.EOS, Offsets.PAD)
            self.inv_mask = tf.cast(tf.equal(self.mask, 0), tf.float32) * tf.constant(-1e4)

            self.A = tf.add(tf.multiply(A, self.mask), self.inv_mask, name="transitions")
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths, self.A)
        else:
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths)
        return tf.reduce_mean(-ll)

    def create_loss(self):

        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float32)
            mask = tf.sign(gold)

            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss()
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        return all_loss

    def __init__(self):
        super(RNNTaggerModelELMoModel, self).__init__()
        pass

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):

        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict['lengths']
        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:

            probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)
            batch_sz, _, label_sz = probv.shape
            start = np.full((batch_sz, 1, label_sz), -1e4)
            start[:, 0, Offsets.GO] = 0
            probv = np.concatenate([start, probv], 1)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl + 1]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                viterbi = viterbi[1:]
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = self.sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    @staticmethod
    def index2word(vocab):

        # Make a vocab list
        vocab_list = [''] * len(vocab)

        for v, i in vocab.items():
            vocab_list[i] = v
        vocab_list[0] = ''

        i2w = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant(vocab_list),
            default_value='',
            name='index2word'
        )

        return i2w

    @staticmethod
    def create(labels, embeddings, **kwargs):

        elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

        word_vec = embeddings['word']
        char_vec = embeddings['char']
        model = RNNTaggerModelELMoModel()
        model.sess = kwargs.get('sess', tf.Session())

        model.mxlen = kwargs.get('maxs', 100)
        model.maxw = kwargs.get('maxw', 100)

        hsz = int(kwargs['hsz'])
        pdrop = kwargs.get('dropout', 0.5)
        model.labels = labels
        model.crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.proj = bool(kwargs.get('proj', False))
        model.feed_input = bool(kwargs.get('feed_input', False))
        model.activation_type = kwargs.get('activation', 'tanh')

        char_dsz = char_vec.dsz
        nc = len(labels)
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, model.mxlen], name="x"))
        model.x_lc = kwargs.get('x_lc', tf.placeholder(tf.int32, [None, model.mxlen], name="x_lc"))

        model.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, model.mxlen, model.maxw], name="xch"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, model.mxlen], name="y"))
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = pdrop
        model.word_vocab = {}
        model.word_vocab = word_vec.vocab
        model.i2w = RNNTaggerModelELMoModel.index2word(model.word_vocab)
        model.char_vocab = char_vec.vocab
        seed = np.random.randint(10e8)

        words = model.i2w.lookup(tf.cast(model.x, dtype=tf.int64))
        # words = tf.Print(words, [words])
        welmo = elmo(
            inputs={
                "tokens": words,
                "sequence_len": model.lengths
            }, signature="tokens", as_dict=True)["elmo"]

        with tf.variable_scope("WordLUT"):
            Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")
            we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
            with tf.control_dependencies([we0]):
                wembed = tf.nn.embedding_lookup(Ww, model.x_lc, name="embeddings")

        Wch = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch")
        ce0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

        word_char = RNNTaggerModelELMoModel.pool_chars(Wch, ce0, char_dsz, kwargs, model)
        nlayers = int(kwargs.get('layers', 1))
        if nlayers > 2:
            raise Exception('Expected 1 or 2 layer stacking only!')

        joint = word_char if word_vec is None else tf.concat(values=[wembed, welmo, word_char], axis=2)
        embedseq = tf.nn.dropout(joint, model.pkeep)

        with tf.variable_scope("brnn1"):

            rnnfwd = lstm_cell(hsz)
            rnnbwd = lstm_cell(hsz)
            rnnout, _ = tf.nn.bidirectional_dynamic_rnn(rnnfwd, rnnbwd, embedseq, sequence_length=model.lengths, dtype=tf.float32)
            rnnout = tf.concat(axis=2, values=rnnout)


        if nlayers == 2:
            with tf.variable_scope("brnn2"):
                rnnfwd = lstm_cell(hsz)
                rnnbwd = lstm_cell(hsz)
                rnn2, _ = tf.nn.bidirectional_dynamic_rnn(rnnfwd, rnnbwd, rnnout, sequence_length=model.lengths, dtype=tf.float32)
                # The output of the BRNN function needs to be joined on the H axis
                rnn2 = tf.concat(axis=2, values=rnn2)
                rnnout = tf.nn.dropout(tf.concat(axis=2, values=[rnnout, rnn2]), model.pkeep)

        with tf.variable_scope("output"):

            # Converts seq to tensor, back to (B,T,W)
            hout = rnnout.get_shape()[-1]
            # Flatten from [B x T x H] - > [BT x H]
            rnnout_bt_x_h = tf.reshape(rnnout, [-1, hout])
            init = xavier_initializer(True, seed)

            with tf.contrib.slim.arg_scope([fully_connected], weights_initializer=init):
                if model.proj is True:
                    hidden = tf.nn.dropout(fully_connected(rnnout_bt_x_h, hsz,
                                                           activation_fn=tf_activation(model.activation_type)), model.pkeep)
                    preds = fully_connected(hidden, nc, activation_fn=None, weights_initializer=init)
                else:
                    preds = fully_connected(rnnout_bt_x_h, nc, activation_fn=None, weights_initializer=init)
            model.probs = tf.reshape(preds, [-1, model.mxlen, nc])
            model.best = tf.argmax(model.probs, 2)
        return model

    @staticmethod
    def pool_chars(Wch, ce0, char_dsz, kwargs, model):
        wsz = kwargs.get('wsz', 30)
        filtsz = kwargs.get('cfiltsz')
        with tf.variable_scope("Chars2Word"):
            with tf.control_dependencies([ce0]):
                rnnchar_bt_x_w = tf.reshape(model.xch, [-1, model.maxw])
                mxfiltsz = np.max(filtsz)
                halffiltsz = mxfiltsz // 2
                zeropad = tf.pad(rnnchar_bt_x_w, [[0, 0], [halffiltsz, halffiltsz]], "CONSTANT")
                cembed = tf.nn.embedding_lookup(Wch, zeropad, name="embeddings")
                cmot, num_filts = char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz,
                                                 activation_fn=tf_activation(model.activation_type))
                word_char = tf.reshape(cmot, [-1, model.mxlen, num_filts])

        return word_char


def create_model(labels, embeddings, **kwargs):
    return RNNTaggerModelELMoModel.create(labels, embeddings, **kwargs)


def load_model(modelname, **kwargs):
    return RNNTaggerModelELMoModel.load(modelname, **kwargs)
