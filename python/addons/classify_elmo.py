import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.utils import fill_y
from baseline.model import ClassifierModel, load_classifier_model, create_classifier_model
import tensorflow_hub as hub


class ELMoClassifierModel(ClassifierModel):
    """
    Convolutional model following baseline, but here use ELMo embeddings
    """
    def __init__(self):
        """Base
        """
        super(ELMoClassifierModel, self).__init__()

    def save(self, outfile):
        """Save a word-based model, along with the label and word indices

        :param outfile:
        :return:
        """
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

    def create_loss(self):

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def predict(self, batch_dict):
        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output

        :param x: The `x` tensor of input (`BxT`)
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """
        x = batch_dict['x']
        lengths = batch_dict['lengths']
        feed_dict = {self.x: x, self.lengths: lengths, self.pkeep: 1.0}
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, train=False):
        x = batch_dict['x']
        y = batch_dict['y']
        lengths = batch_dict['lengths']
        pkeep = 1.0 - self.pdrop_value if train else 1
        return {self.x: x, self.lengths: lengths, self.y: fill_y(len(self.labels), y), self.pkeep: pkeep}

    def get_labels(self):
        return self.labels

    def get_vocab(self, name):
        return self.vocab if name == 'word' else None

    @classmethod
    def load(cls, basename, **kwargs):
        """Reload the model from a graph file and a checkpoint

        The model that is loaded is independent of the pooling and stacking layers, making this class reusable
        by sub-classes.

        :param basename: The base directory to load from
        :param kwargs: See below

        :Keyword Arguments:
        * *session* -- An optional tensorflow session.  If not passed, a new session is
            created

        :return: A restored model
        """
        sess = kwargs.get('session', tf.Session())
        model = cls()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name + ".model"})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')
            sess.run(tf.get_default_graph().get_operation_by_name('index2word/table_init'))

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        with open(basename + '.vocab', 'r') as f:
            model.vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        model.sess = sess
        return model

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

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        sess = kwargs.get('sess', tf.Session())
        finetune = bool(kwargs.get('finetune', True))
        mxlen = int(kwargs.get('mxlen', 100))
        w2v = embeddings['word']
        model = cls()
        dsz = w2v.dsz
        model.labels = labels
        nc = len(labels)
        model.vocab = w2v.vocab
        model.i2w = ELMoClassifierModel.index2word(model.vocab)
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, mxlen], name="x"))
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        # I wish there was an elegant way to avoid this
        filtsz = kwargs.get('filtsz', [3, 4, 5]) if kwargs.get('model_type', 'default') == 'default' else 0

        mxfiltsz = np.max(filtsz)
        halffiltsz = mxfiltsz // 2
        print(filtsz, halffiltsz)
        seed = np.random.randint(10e8)
        init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        # Use pre-trained embeddings from word2vec
        with tf.variable_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name="W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
            with tf.control_dependencies([e0]):
                # Zeropad out the word ids in the sentence to half the max
                # filter size, to make a wide convolution.  This way we
                # don't have to explicitly pad the x data upfront
                zeropad = tf.pad(model.x, [[0,0], [halffiltsz, halffiltsz]], "CONSTANT")
                word_embeddings = tf.nn.embedding_lookup(W, zeropad)

        words = model.i2w.lookup(tf.cast(model.x, dtype=tf.int64))
        # words = tf.Print(words, [words])
        welmo = elmo(
            inputs={
                "tokens": words,
                "sequence_len": model.lengths
            }, signature="tokens", as_dict=True)["elmo"]

        joint = tf.concat(values=[word_embeddings, welmo], axis=2)
        pooled = model.pool(joint, dsz + 1024, init, **kwargs)
        # combine = highway_conns(combine, wsz_all, 1)

        stacked = model.stacked(pooled, init, **kwargs)

        # For fully connected layers, use xavier (glorot) transform
        with tf.contrib.slim.arg_scope(
                [fully_connected],
                weights_initializer=xavier_init):

            with tf.variable_scope("output"):
                model.logits = tf.identity(fully_connected(stacked, nc, activation_fn=None), name="logits")
                model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        return model

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling

        :param word_embeddings: The word embeddings, which are inputs here
        :param dsz: The depth of the word embeddings
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use


        :return:
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        expanded = tf.expand_dims(word_embeddings, -1)
        mots = []

        for i, fsz in enumerate(filtsz):
            with tf.variable_scope('cmot-%s' % fsz):

                kernel_shape = [fsz, dsz, 1, cmotsz]

                W = tf.get_variable("W", kernel_shape, initializer=init)
                b = tf.get_variable("b", [cmotsz], initializer=tf.constant_initializer(0.0))

                conv = tf.nn.conv2d(expanded,
                                    W, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

                mot = tf.reduce_max(activation, [1], keep_dims=True)
                # Add back in the dropout
                mots.append(mot)

        wsz_all = cmotsz * len(mots)
        combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])
        # Definitely drop out
        with tf.name_scope("dropout"):
            combine = tf.nn.dropout(combine, self.pkeep)
        return combine

    def stacked(self, pooled, init, **kwargs):
        """This takes the pooled layer and puts it through a stack of (presumably) fully-connected layers.
        Not all implementations utilize this

        :param pooled: The output of the pooling layer
        :param init: The tensorflow initializer to use for these methods
        :return: The final representation of the stacking.  By default, do none
        """
        return pooled


def create_model(word_embeddings, labels, **kwargs):
    classifier = ELMoClassifierModel.create(word_embeddings, labels, **kwargs)
    return classifier

def load_model(modelname, **kwargs):
    return ELMoClassifierModel.load(modelname, **kwargs)
