import tensorflow as tf
import numpy as np
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import json
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.utils import fill_y, listify
from baseline.model import Classifier, load_classifier_model, create_classifier_model
from baseline.tf.tfy import (stacked_lstm,
                             parallel_conv,
                             get_vocab_file_suffixes)

from baseline.tf.embeddings import (TensorFlowEmbeddings,
                                    TensorFlowCharConvEmbeddings,
                                    TensorFlowTokenEmbeddings,
                                    tf_embeddings)
from baseline.version import __version__
import os
from baseline.utils import zip_model, unzip_model
import copy


class ClassifyParallelModel(Classifier):

    def __init__(self, create_fn, embeddings, labels, **kwargs):
        super(ClassifyParallelModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus', -1)
        # If the gpu ID is set to -1, use CUDA_VISIBLE_DEVICES to figure it out
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        print('Num GPUs', gpus)

        self.labels = labels
        nc = len(labels)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()
        for key in embeddings.keys():
            Type = TensorFlowCharConvEmbeddings if key == 'char' else TensorFlowTokenEmbeddings
            if isinstance(embeddings[key], TensorFlowEmbeddings):
                Type = embeddings[key].__class__

            self.parallel_params[key] = kwargs.get(key, Type.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus)

        self.lengths_key = kwargs.get('lengths_key')

        if self.lengths_key is not None:
            # This allows user to short-hand the field to use
            if not self.lengths_key.endswith('_lengths'):
                self.lengths_key += '_lengths'
            self.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths_parallel"))
            lengths_splits = tf.split(self.lengths, gpus)
            split_operations['lengths'] = lengths_splits

        else:
            self.lengths = None

        # This only exists to make exporting easier
        self.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y_parallel"))
        self.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths_parallel"))
        self.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        self.pdrop_value = kwargs.get('dropout', 0.5)

        y_splits = tf.split(self.y, gpus)
        split_operations['y'] = y_splits
        losses = []
        self.labels = labels

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) 
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            self.inference = create_fn(embeddings, labels, sess=sess, **kwargs)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                kwargs_single = copy.deepcopy(kwargs)
                kwargs_single['sess'] = sess
                kwargs_single['pkeep'] = self.pkeep

                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                replica = create_fn(embeddings, labels, **kwargs_single)
                self.replicas.append(replica)
                loss_op = replica.create_loss()
                losses.append(loss_op)

        self.loss = tf.reduce_mean(tf.stack(losses))

        self.sess = sess
        self.best = self.inference.best

    def create_loss(self):
        return self.loss

    def create_test_loss(self):
        return self.inference.create_test_loss()

    def save(self, model_base):
        return self.inference.save(model_base)

    def set_saver(self, saver):
        self.inference.saver = saver
        self.saver = saver

    def make_input(self, batch_dict, do_dropout=False):
        if do_dropout is False:
            return self.inference.make_input(batch_dict)

        y = batch_dict.get('y', None)
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.parallel_params.keys():
            feed_dict["{}_parallel:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = fill_y(len(self.labels), y)
        return feed_dict


class ClassifierBase(Classifier):
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
        super(ClassifierBase, self).__init__()

    def set_saver(self, saver):
        self.saver = saver

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"version": __version__, "embeddings": list(self.embeddings.keys())}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)

        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}.vocab'.format(key))
        state_file = basename + '.state'
        # Backwards compat for now
        if not os.path.exists(state_file):
            return

    def save(self, basename, **kwargs):
        self.save_md(basename)
        self.save_values(basename)

    def create_test_loss(self):
        with tf.name_scope("test_loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.y, "float"))
            all_loss = tf.reduce_mean(loss)
        return all_loss

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
        
        :param batch_dict: (``dict``) contains `x` tensor of input (`BxT`)
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """
        feed_dict = self.make_input(batch_dict)
        probs = self.sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, do_dropout=False):
        y = batch_dict.get('y', None)

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        if self.lengths_key is not None:
            feed_dict[self.lengths] = batch_dict[self.lengths_key]

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
        return self.vocab.get(name)

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
        basename = unzip_model(basename)
        sess = kwargs.get('session', kwargs.get('sess', tf.Session()))
        model = cls()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename

        vocab_suffixes = get_vocab_file_suffixes(basename)
        for ty in vocab_suffixes:
            vocab_file = '{}-{}.vocab'.format(basename, ty)
            print('Reading {}'.format(vocab_file))
            model.embeddings[ty] = TensorFlowTokenEmbeddings.load(vocab_file)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            try:
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            except:
                # Backwards compat
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name + ".model"})

            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            try:
                model.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            except:
                model.xch = None
            try:
                model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            except:
                model.lengths = None
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')
            model.logits = tf.get_default_graph().get_tensor_by_name('output/logits:0')
        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

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

        gpus = kwargs.get('gpus')
        # If we are parallelized, we will use the wrapper object ClassifyParallelModel and this creation function
        if gpus is not None:
            return ClassifyParallelModel(cls.create, embeddings, labels, **kwargs)
        sess = kwargs.get('sess', tf.Session())

        model = cls()
        model.embeddings = dict()
        for key in embeddings.keys():
            DefaultType = TensorFlowCharConvEmbeddings if key == 'char' else TensorFlowTokenEmbeddings
            model.embeddings[key] = tf_embeddings(embeddings[key], key, DefaultType=DefaultType, **kwargs)

        model.lengths_key = kwargs.get('lengths_key')

        if model.lengths_key is not None:
            # This allows user to short-hand the field to use
            if not model.lengths_key.endswith('_lengths'):
                model.lengths_key += '_lengths'
            model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        else:
            model.lengths = None

        model.labels = labels
        nc = len(labels)
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        # This only exists to make exporting easier
        model.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        # This only exists to make exporting easier

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            seed = np.random.randint(10e8)
            init = tf.random_uniform_initializer(-0.05, 0.05, dtype=tf.float32, seed=seed)
            xavier_init = xavier_initializer(True, seed)

            all_embeddings_out = []
            for embedding in model.embeddings.values():
                embeddings_out = embedding.encode()
                all_embeddings_out += [embeddings_out]

            word_embeddings = tf.concat(values=all_embeddings_out, axis=2)
            input_sz = word_embeddings.shape[-1]
            pooled = model.pool(word_embeddings, input_sz, init, **kwargs)
            stacked = model.stacked(pooled, init, **kwargs)

            # For fully connected layers, use xavier (glorot) transform
            with tf.contrib.slim.arg_scope(
                    [fully_connected],
                    weights_initializer=xavier_init):
                with tf.variable_scope("output"):
                    model.logits = tf.identity(fully_connected(stacked, nc, activation_fn=None), name="logits")
                    model.best = tf.argmax(model.logits, 1, name="best")
        model.sess = sess
        # writer = tf.summary.FileWriter('blah', sess.graph)
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


class ConvModel(ClassifierBase):
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

        combine, _ = parallel_conv(word_embeddings, filtsz, dsz, cmotsz)
        # Definitely drop out
        with tf.name_scope("dropout"):
            combine = tf.nn.dropout(combine, self.pkeep)
        return combine


class LSTMModel(ClassifierBase):
    """A simple single-directional single-layer LSTM. No layer-stacking.
    
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

        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        nlayers = int(kwargs.get('layers', 1))

        if rnntype == 'blstm':
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            rnnbwd = stacked_lstm(hsz, self.pkeep, nlayers)
            ((_, _), (fw_final_state, bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(rnnfwd,
                                                                                         rnnbwd,
                                                                                         word_embeddings,
                                                                                         sequence_length=self.lengths,
                                                                                         dtype=tf.float32)
            # The output of the BRNN function needs to be joined on the H axis
            output_state = fw_final_state[-1].h + bw_final_state[-1].h
            out_hsz = hsz

        else:
            rnnfwd = stacked_lstm(hsz, self.pkeep, nlayers)
            (_, (output_state)) = tf.nn.dynamic_rnn(rnnfwd, word_embeddings, sequence_length=self.lengths, dtype=tf.float32)
            output_state = output_state[-1].h
            out_hsz = hsz

        combine = tf.reshape(output_state, [-1, out_hsz])
        return combine


class NBowBase(ClassifierBase):
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
        return super(NBowBase, self).stacked(pooled, init, **kwargs)


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
        return tf.reduce_mean(word_embeddings, 1, keepdims=False)


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
        return tf.reduce_max(word_embeddings, 1, keepdims=False)


class CompositePoolingModel(ClassifierBase):

    def __init__(self):
        super(CompositePoolingModel, self).__init__()

    def pool(self, word_embeddings, dsz, init, **kwargs):
        SubModels = [eval(model) for model in kwargs.get('sub')]
        pooling = []
        for SubClass in SubModels:
            pooling += [SubClass.pool(self, word_embeddings, dsz, init, **kwargs)]
        return tf.concat(pooling, -1)


BASELINE_CLASSIFICATION_MODELS = {
    'default': ConvModel.create,
    'lstm': LSTMModel.create,
    'nbow': NBowModel.create,
    'nbowmax': NBowMaxModel.create,
    'composite': CompositePoolingModel.create
}
BASELINE_CLASSIFICATION_LOADERS = {
    'default': ConvModel.load,
    'lstm': LSTMModel.load,
    'nbow': NBowModel.load,
    'nbowmax': NBowMaxModel.load
}


def create_model(embeddings, labels, **kwargs):
    return create_classifier_model(BASELINE_CLASSIFICATION_MODELS, embeddings, labels, **kwargs)


def load_model(outname, **kwargs):
    return load_classifier_model(BASELINE_CLASSIFICATION_LOADERS, outname, **kwargs)
