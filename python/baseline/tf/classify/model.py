import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import convolution2d, fully_connected, flatten, xavier_initializer
from baseline.utils import fill_y, create_user_model, load_user_model
from baseline.model import Classifier


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

        # Create parallel filter operations of different sizes
        with tf.contrib.slim.arg_scope(
                [convolution2d, fully_connected],
                weights_initializer=init,
                biases_initializer=tf.constant_initializer(0)):

            for i, fsz in enumerate(filtsz):
                with tf.name_scope('cmot-%s' % fsz) as scope:
                    conv = convolution2d(expanded, cmotsz, [fsz, dsz], [1, 1], padding='VALID', scope=scope)
                    # First dim is batch, second dim is time, third dim is feature map
                    # Max over time pooling, 2 calls below are equivalent
                    mot = tf.reduce_max(conv, [1], keep_dims=True)
                mots.append(mot)

            combine = flatten(tf.concat(values=mots, axis=3))

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


def create_model(w2v, labels, **kwargs):
    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        return ConvModel.create(w2v, labels, **kwargs)

    model = create_user_model(model_type, w2v, labels, **kwargs)
    return model

@staticmethod
def load_model(outname, **kwargs):

    model_type = kwargs.get('model_type', 'default')
    if model_type == 'default':
        return ConvModel.load(outname, **kwargs)
    return load_user_model(model_type, outname, **kwargs)