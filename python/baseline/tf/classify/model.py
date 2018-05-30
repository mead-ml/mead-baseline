import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.utils import fill_y, listify
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.tf.tfy import lstm_cell_w_dropout, parallel_conv, get_vocab_file_suffixes, pool_chars
from baseline.version import __version__
import os


class WordClassifierBase(Classifier):
    """Base for all baseline implementations of word-based classifiers
    
    This class provides a loose skeleton around which the baseline models (currently all word-based)
    are built.  This essentially consists of dividing up the network into a logical separation between "pooling",
    or the conversion of temporal data to a fixed representation, and "stacking" layers, which are (optional)
    fully-connected layers below, finally followed with a penultimate layer that is projected to the output space.
    
    For instance, the baseline convolutional and LSTM models implement pooling as CMOT, and LSTM last time
    respectively, whereas, neural bag-of-words (NBoW) do simple max or mean pooling followed by multiple fully-
    connected layers.
    
    """
    def __init__(self):
        """Base
        """
        super(WordClassifierBase, self).__init__()

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"mxlen": self.mxlen, "version": __version__}
        if self.mxwlen is not None:
            state["mxwlen"] = self.mxwlen
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        for key in self.vocab.keys():
            with open(basename + '-{}.vocab'.format(key), 'w') as f:
                json.dump(self.vocab[key], f)

    def load_md(self, basename):

        state_file = basename + '.state'
        # Backwards compat for now
        if not os.path.exists(state_file):
            return

        with open(state_file, 'w') as f:
            state = json.load(f)
            self.mxlen = state.get('mxlen')
            self.mxwlen = state.get('mxwlen')

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def create_loss(self):
        """The loss function is currently provided here, although this is not a great place for it
        as it provides a coupling between the model and its loss function.  Just here for convenience at the moment.
        
        :return: 
        """
        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

    def classify(self, batch_dict):
        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output
        
        :param x: The `x` tensor of input (`BxT`)
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """
        feed_dict = self.make_input(self, batch_dict)
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, do_dropout=False):
        x = batch_dict['x']
        y = batch_dict.get('y', None)
        xch = batch_dict.get('xch')
        lengths = batch_dict.get('lengths')
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.x: x, self.pkeep: pkeep}
        if xch is not None and self.xch is not None:
            feed_dict[self.xch] = xch
        #if lengths is not None and self.lengths is not None:
        #    feed_dict[self.lengths] = lengths
        if y is not None:
            feed_dict[self.y] = fill_y(len(self.labels), y)
        return feed_dict

    def get_labels(self):
        """Get the string labels back
        
        :return: labels
        """
        return self.labels

    def get_vocab(self, name='word'):
        """Get the vocab back, as a ``dict`` of ``str`` keys mapped to ``int`` values
        
        :return: A ``dict`` of words mapped to indices
        """
        return self.vocab[name]

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

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: basename + '.model'})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            try:
                model.xch = tf.get_default_graph().has_tensor_by_name('xch:0')
            except:
                model.xch = None
            try:
                model.lengths = tf.get_default_graph().has_tensor_by_name('lengths:0')
            except:
                model.lengths = None
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        model.vocab = {}

        # Backwards compat
        if os.path.exists(basename + '.vocab'):
            with open(basename + '.vocab', 'r') as f:
                model.vocab['word'] = json.load(f)
        # Grep for all features
        else:
            vocab_suffixes = get_vocab_file_suffixes(basename)
            for ty in vocab_suffixes:
                vocab_file = '{}-{}.vocab'.format(basename, ty)
            print('Reading {}', vocab_file)
            with open(vocab_file, 'r') as f:
                model.vocab[ty] = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        model.sess = sess
        model.load_md(basename)
        return model

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        """The main method for creating all :class:`WordBasedModel` types.
        
        This method instantiates a model with pooling and optional stacking layers.
        Many of the arguments provided are reused by each implementation, but some sub-classes need more
        information in order to properly initialize.  For this reason, the full list of keyword args are passed
        to the :method:`pool` and :method:`stacked` methods.
        
        :param embeddings: This is a dictionary of embeddings, mapped to their numerical indices in the lookup table
        :param labels: This is a list of the `str` labels
        :param kwargs: See below
        
        :Keyword Arguments:
        * *model_type* -- The string name for the model (defaults to `default`)
        * *session* -- An optional tensorflow session.  If not passed, a new session is
            created
        * *finetune* -- Are we doing fine-tuning of word embeddings (defaults to `True`)
        * *mxlen* -- The maximum signal (`x` tensor temporal) length (defaults to `100`)
        * *dropout* -- This indicates how much dropout should be applied to the model when training.
        * *pkeep* -- By default, this is a `tf.placeholder`, but it can be passed in as part of a sub-graph.
            This is useful for exporting tensorflow models or potentially for using input tf queues
        * *x* -- By default, this is a `tf.placeholder`, but it can be optionally passed as part of a sub-graph.
        * *y* -- By default, this is a `tf.placeholder`, but it can be optionally passed as part of a sub-graph.
        * *filtsz* -- This is actually a top-level param due to an unfortunate coupling between the pooling layer
            and the input, which, for convolution, requires input padding.
        
        :return: A fully-initialized tensorflow classifier 
        """
        sess = kwargs.get('sess', tf.Session())
        finetune = bool(kwargs.get('finetune', True))
        w2v = embeddings['word']
        c2v = embeddings.get('char')

        model = cls()
        word_dsz = w2v.dsz
        wchsz = 0
        model.labels = labels
        nc = len(labels)
        model.vocab = w2v.vocab
        model.mxlen = int(kwargs.get('mxlen', 100))
        model.mxwlen = None
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, model.mxlen], name="x"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        model.xch = None
        seed = np.random.randint(10e8)
        init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        # Use pre-trained embeddings from word2vec
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name="W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_dsz]))
            with tf.control_dependencies([e0]):
                word_embeddings = tf.nn.embedding_lookup(W, model.x)

        if c2v is not None:
            model.mxwlen = int(kwargs.get('mxwlen', 40))
            model.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, model.mxlen, model.mxwlen]))
            char_dsz = c2v.dsz
            with tf.name_scope("CharLUT"):
                Wch = tf.Variable(tf.constant(c2v.weights, dtype=tf.float32), name="Wch", trainable=True)
                ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))
                char_comp, wchsz = pool_chars(model.xch, Wch, ech0, char_dsz, **kwargs)
                word_embeddings = tf.concat(values=[word_embeddings, char_comp], axis=2)

        input_sz = word_dsz + wchsz
        pooled = model.pool(word_embeddings, input_sz, init, **kwargs)
        stacked = model.stacked(pooled, init, **kwargs)

        # For fully connected layers, use xavier (glorot) transform
        with tf.contrib.slim.arg_scope(
                [fully_connected],
                weights_initializer=xavier_init):
            with tf.name_scope("output"):
                model.logits = tf.identity(fully_connected(stacked, nc, activation_fn=None), name="logits")
                model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        return model

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """This method performs a transformation between a temporal signal and a fixed representation
        
        :param word_embeddings: The output of the embedded lookup, which is the starting point for this operation
        :param dsz: The depth of the embeddings
        :param init: The tensorflow initializer to use for these methods
        :param kwargs: Model-specific arguments
        :return: A fixed representation of the data
        """
        pass

    def stacked(self, pooled, init, **kwargs):
        """Stack 1 or more hidden layers, optionally (forming an MLP)

        :param pooled: The fixed representation of the model
        :param init: The tensorflow initializer
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)

        :return: The final layer
        """

        hszs = listify(kwargs.get('hsz', []))
        if len(hszs) == 0:
            return pooled

        in_layer = pooled
        for i, hsz in enumerate(hszs):
            with tf.variable_scope('fc-{}'.format(i)):
                with tf.contrib.slim.arg_scope(
                        [fully_connected],
                        weights_initializer=init):
                    fc = fully_connected(in_layer, hsz, activation_fn=tf.nn.relu)
                    in_layer = tf.nn.dropout(fc, self.pkeep)
        return in_layer


class ConvModel(WordClassifierBase):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width
    
    """
    def __init__(self):
        """Constructor 
        """
        super(ConvModel, self).__init__()

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

        combine = parallel_conv(word_embeddings, filtsz, dsz, cmotsz)
        # Definitely drop out
        with tf.name_scope("dropout"):
            combine = tf.nn.dropout(combine, self.pkeep)
        return combine


class LSTMModel(WordClassifierBase):
    """A simple single-directional single-layer LSTM. No layer-stacking. Pad sequences left for this model
    
    """
    def __init__(self):
        super(LSTMModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """LSTM with dropout yielding a final-state as output
        
        :param word_embeddings: The input word embeddings
        :param dsz: The input word embedding depth
        :param init: The tensorflow initializer to use (currently ignored)
        :param kwargs: See below
        
        :Keyword Arguments:
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)
        * *cmotsz* -- (``int``) An alias for `hsz`
        
        :return: 
        """
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        if type(hsz) is list:
            hsz = hsz[0]
        char_rnnfwd = lstm_cell_w_dropout(hsz, self.pkeep)
        rnnout, final_state = tf.nn.dynamic_rnn(char_rnnfwd, word_embeddings, dtype=tf.float32)

        output_state = final_state.h
        combine = tf.reshape(output_state, [-1, hsz])
        return combine


class NBowBase(WordClassifierBase):
    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """
    def __init__(self):
        super(NBowBase, self).__init__()

    def stacked(self, pooled, init, **kwargs):
        """Force at least one hidden layer here

        :param pooled:
        :param init:
        :param kwargs:
        :return:
        """
        kwargs['hsz'] = kwargs.get('hsz', [100])
        return super(NBowBase, self).stacked()


class NBowModel(NBowBase):
    """Neural Bag-of-Words average pooling (standard) model"""
    def __init__(self):
        super(NBowModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do average pooling on input embeddings, yielding a `dsz` output layer
        
        :param word_embeddings: The word embedding input
        :param dsz: The word embedding depth
        :param init: The tensorflow initializer
        :param kwargs: None
        :return: The average pooling representation
        """
        return tf.reduce_mean(word_embeddings, 1, keep_dims=False)


class NBowMaxModel(NBowBase):
    """Max-pooling model for Neural Bag-of-Words.  Sometimes does better than avg pooling
    """
    def __init__(self):
        super(NBowMaxModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        """Do max pooling on input embeddings, yielding a `dsz` output layer
        
        :param word_embeddings: The word embedding input
        :param dsz: The word embedding depth
        :param init: The tensorflow initializer
        :param kwargs: None
        :return: The max pooling representation
        """
        return tf.reduce_max(word_embeddings, 1, keep_dims=False)

BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)
