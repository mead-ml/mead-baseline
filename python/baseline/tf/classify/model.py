import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.utils import fill_y
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.tf.tfy import lstm_cell_w_dropout, parallel_conv


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
        x = batch_dict['x']
        feed_dict = {self.x: x, self.pkeep: 1.0}
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, do_dropout=False):
        """Convert from an input of x and y tensors to a `feed_dict`
        
        :param x: Input tensor `x` (`BxT`)
        :param y: Input tensor `y` (`B`)
        :param do_dropout: Defaults to off.  If its on, use the dropout value provided during model construction
        :return: A `feed_dict`
        """

        x = batch_dict['x']
        y = batch_dict['y']
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1
        return {self.x: x, self.y: fill_y(len(self.labels), y), self.pkeep: pkeep}

    def get_labels(self):
        """Get the string labels back
        
        :return: labels
        """
        return self.labels

    def get_vocab(self):
        """Get the vocab back, as a ``dict`` of ``str`` keys mapped to ``int`` values
        
        :return: A ``dict`` of words mapped to indices
        """
        return self.vocab

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
        mxlen = int(kwargs.get('mxlen', 100))
        w2v = embeddings['word']
        model = cls()
        dsz = w2v.dsz
        model.labels = labels
        nc = len(labels)
        model.vocab = w2v.vocab
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, mxlen], name="x"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        # I wish there was an elegant way to avoid this
        filtsz = kwargs.get('filtsz', [3, 4, 5]) if kwargs.get('model_type', 'default') == 'default' else 0

        seed = np.random.randint(10e8)
        init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
        xavier_init = xavier_initializer(True, seed)

        # Use pre-trained embeddings from word2vec
        with tf.name_scope("LUT"):
            W = tf.Variable(tf.constant(w2v.weights, dtype=tf.float32), name="W", trainable=finetune)
            e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
            with tf.control_dependencies([e0]):
                word_embeddings = tf.nn.embedding_lookup(W, model.x)

        pooled = model.pool(word_embeddings, dsz, init, **kwargs)
        # combine = highway_conns(combine, wsz_all, 1)

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
        """This takes the pooled layer and puts it through a stack of (presumably) fully-connected layers.
        Not all implementations utilize this
        
        :param pooled: The output of the pooling layer
        :param init: The tensorflow initializer to use for these methods
        :return: The final representation of the stacking.  By default, do none
        """
        return pooled


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
        hsz = kwargs.get('hsz', kwargs.get('cmotsz', 100))
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
        """To make a neural bag of words, stack 1 or more hidden layers (forming an MLP)
        
        :param pooled: The fixed representation of the model
        :param init: The tensorflow initializer
        :param kwargs: See below
        
        :Keyword Arguments:
        * *layers* -- (``int``) The number of hidden layers (defaults to `1`)
        * *hsz* -- (``int``) The number of hidden units (defaults to `100`)
        * *cmotsz* -- (``int``) An alias for `hsz`
        
        :return: The final layer
        """
        nlayers = kwargs.get('layers', 1)
        hsz = kwargs.get('hsz', kwargs.get('cmotsz', 100))

        in_layer = pooled
        for i in range(nlayers):
            with tf.variable_scope('fc-%s' % i):
                with tf.contrib.slim.arg_scope(
                        [fully_connected],
                        weights_initializer=init):

                    fc = fully_connected(in_layer, hsz, activation_fn=tf.nn.relu)
                    in_layer = tf.nn.dropout(fc, self.pkeep)
        return in_layer


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
