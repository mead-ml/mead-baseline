import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import convolution2d, fully_connected, flatten, xavier_initializer
from baseline.utils import fill_y
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.tf.tfy import lstm_cell_w_dropout


class ConvModel(Classifier):

    def __init__(self):
        super(ConvModel, self).__init()

    def save(self, outfile):
        path_and_file = outfile.split('/')
        outdir = '/'.join(path_and_file[:-1])
        base = path_and_file[-1]
        basename = outdir + '/' + base
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(self.sess, basename + '.model')

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    @staticmethod
    def load(basename, **kwargs):

        sess = kwargs.get('session', tf.Session())
        model = ConvModel()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: basename + '.model'})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        model.sess = sess
        return model

    def __init__(self):
        pass

    def create_loss(self):

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def classify(self, x):
        feed_dict = {self.x: x, self.pkeep: 1.0}
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def ex2dict(self, x, y, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1
        return {self.x: x, self.y: fill_y(len(self.labels), y), self.pkeep: pkeep}

    @staticmethod
    def create(w2v, labels, **kwargs):
        sess = kwargs.get('sess', tf.Session())
        finetune = bool(kwargs.get('finetune', True))
        mxlen = int(kwargs.get('mxlen', 100))
        filtsz = kwargs['filtsz']
        cmotsz = kwargs['cmotsz']
        model = ConvModel()
        dsz = w2v.dsz
        model.labels = labels
        nc = len(labels)
        model.vocab = w2v.vocab
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, mxlen], name="x"))
        model.y = tf.placeholder(tf.int32, [None, nc], name="y")
        mxfiltsz = np.max(filtsz)
        halffiltsz = mxfiltsz // 2

        # Use pre-trained embeddings from word2vec
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name="W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
            with tf.control_dependencies([e0]):
                # Zeropad out the word ids in the sentence to half the max
                # filter size, to make a wide convolution.  This way we
                # don't have to explicitly pad the x data upfront
                zeropad = tf.pad(model.x, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT") 
                lut = tf.nn.embedding_lookup(W, zeropad)
                expanded = tf.expand_dims(lut, -1)

        mots = []

        seed = np.random.randint(10e8)
        init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        for i, fsz in enumerate(filtsz):
            with tf.variable_scope('cmot-%s' % fsz):

                kernel_shape = [fsz, dsz, 1, cmotsz]

                # Weight tying
                W = tf.get_variable("W", kernel_shape, initializer=init)
                b = tf.get_variable("b", [cmotsz], initializer=tf.constant_initializer(0.0))

                conv = tf.nn.conv2d(expanded,
                                    W, strides=[1,1,1,1],
                                    padding="VALID", name="conv")

                activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

                mot = tf.reduce_max(activation, [1], keep_dims=True)
                # Add back in the dropout
                mots.append(mot)

        wsz_all = cmotsz * len(mots)
        combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])
        # combine = highway_conns(combine, wsz_all, 1)

        # Definitely drop out
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(combine, model.pkeep)

            # For fully connected layers, use xavier (glorot) transform
        with tf.contrib.slim.arg_scope(
                [fully_connected],
                weights_initializer=xavier_init):

            with tf.name_scope("output"):
                model.logits = tf.identity(fully_connected(drop, nc, activation_fn=None), name="logits")
                model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        return model

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab



class LSTMModel(Classifier):

    def __init__(self):
        super(LSTMModel, self).__init()

    def save(self, outfile):
        path_and_file = outfile.split('/')
        outdir = '/'.join(path_and_file[:-1])
        base = path_and_file[-1]
        basename = outdir + '/' + base
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(self.sess, basename + '.model')

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        with open(basename + '.vocab', 'w') as f:
            json.dump(self.vocab, f)

    @staticmethod
    def load(basename, **kwargs):

        sess = kwargs.get('session', tf.Session())
        model = LSTMModel()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: basename + '.model'})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        model.sess = sess
        return model

    def __init__(self):
        pass

    def create_loss(self):

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def classify(self, x):
        feed_dict = {self.x: x, self.pkeep: 1.0}
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def ex2dict(self, x, y, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1
        return {self.x: x, self.y: fill_y(len(self.labels), y), self.pkeep: pkeep}

    @staticmethod
    def create(w2v, labels, **kwargs):
        sess = kwargs.get('sess', tf.Session())
        finetune = bool(kwargs.get('finetune', True))
        mxlen = int(kwargs.get('mxlen', 100))
        filtsz = kwargs['filtsz']
        hsz = kwargs.get('hsz', kwargs.get('cmotsz', 100))
        model = ConvModel()
        dsz = w2v.dsz
        model.labels = labels
        nc = len(labels)
        model.vocab = w2v.vocab
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, mxlen], name="x"))
        model.y = tf.placeholder(tf.int32, [None, nc], name="y")
        mxfiltsz = np.max(filtsz)
        halffiltsz = mxfiltsz // 2

        # Use pre-trained embeddings from word2vec
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name="W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
            with tf.control_dependencies([e0]):
                embed = tf.nn.embedding_lookup(W, model.x)

        seed = np.random.randint(10e8)
        #  init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        char_rnnfwd = lstm_cell_w_dropout(hsz, model.pkeep)
        rnnout, final_state = tf.nn.dynamic_rnn(char_rnnfwd, embed, dtype=tf.float32)

        output_state = final_state.h
        combine = tf.reshape(output_state, [-1, hsz])
        # combine = highway_conns(combine, wsz_all, 1)

        # Definitely drop out
        with tf.name_scope("dropout"):
            drop = tf.nn.dropout(combine, model.pkeep)

            # For fully connected layers, use xavier (glorot) transform
        with tf.contrib.slim.arg_scope(
                [fully_connected],
                weights_initializer=xavier_init):

            with tf.name_scope("output"):
                model.logits = tf.identity(fully_connected(drop, nc, activation_fn=None), name="logits")
                model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        return model

    def get_labels(self):
        return self.labels

    def get_vocab(self):
        return self.vocab

BASELINE_CLASSIFICATION_MODELS = {'default': ConvModel.create, 'lstm': LSTMModel.create}
BASELINE_CLASSIFICATION_LOADERS = {'default': ConvModel.load, 'lstm': LSTMModel.load}


def create_model(w2v, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, w2v, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)