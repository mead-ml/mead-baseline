import os
import copy
import logging
from itertools import chain
import tensorflow as tf

from baseline.tf.embeddings import *
from eight_mile.tf.layers import *
from baseline.version import __version__

from baseline.utils import (
    fill_y,
    listify,
    ls_props,
    read_json,
    write_json,
    MAGIC_VARS,
)
from baseline.model import ClassifierModel, register_model
from baseline.tf.tfy import (
    TRAIN_FLAG,
    stacked_lstm,
    parallel_conv,
    reload_embeddings,
    new_placeholder_dict,
    tf_device_wrapper,
    create_session
)



logger = logging.getLogger('baseline')


class ClassifierModelBase(ClassifierModel):
    """Base for all baseline implementations of token-based classifiers

    This class provides a loose skeleton around which the baseline models
    are built.  This essentially consists of dividing up the network into a logical separation between "embedding",
    or composition of lookup tables to build a vector representation of a temporal input, "pooling",
    or the conversion of temporal data to a fixed representation, and "stacking" layers, which are (optional)
    fully-connected layers below, followed by a projection to output space and a softmax

    For instance, the baseline convolutional and LSTM models implement pooling as CMOT, and LSTM last time
    respectively, whereas, neural bag-of-words (NBoW) do simple max or mean pooling followed by multiple fully-
    connected layers.

    """
    def __init__(self):
        """Base
        """
        super(ClassifierModelBase, self).__init__()
        self._unserializable = []

    def set_saver(self, saver):
        self.saver = saver

    def save_values(self, basename):
        """Save tensor files out

        :param basename: Base name of model
        :return:
        """
        self.saver.save(self.sess, basename)

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """
        write_json(self._state, '{}.state'.format(basename))
        write_json(self.labels, '{}.labels'.format(basename))
        for key, embedding in self.embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

    def _record_state(self, **kwargs):
        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(
            self._unserializable,
            MAGIC_VARS,
            self.embeddings.keys(),
            (f'{k}_lengths' for k in self.embeddings.keys())
        ))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
        })

    def save(self, basename, **kwargs):
        """Save meta-data and actual data for a model

        :param basename: (``str``) The model basename
        :param kwargs:
        :return: None
        """
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

    def predict(self, batch_dict):
        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output. You can use this method directly if you have vector input, or you can use the `ClassifierService`
        which can convert directly from text durign its `transform`.  That method calls this one underneath.

        :param batch_dict: (``dict``) Contains any inputs to embeddings for this model
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """
        feed_dict = self.make_input(batch_dict)
        probs = self.sess.run(self.probs, feed_dict)
        results = []
        batchsz = probs.shape[0]
        for b in range(batchsz):
            outcomes = [(self.labels[id_i], prob_i) for id_i, prob_i in enumerate(probs[b])]
            results.append(outcomes)
        return results

    def make_input(self, batch_dict, train=False):
        """Transform a `batch_dict` into a TensorFlow `feed_dict`

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :param train: (``bool``) Are we training.  Defaults to False
        :return:
        """
        y = batch_dict.get('y', None)
        feed_dict = new_placeholder_dict(train)

        for k in self.embeddings.keys():
            feed_dict["{}:0".format(k)] = batch_dict[k]

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

    @classmethod
    @tf_device_wrapper
    def load(cls, basename, **kwargs):
        """Reload the model from a graph file and a checkpoint

        The model that is loaded is independent of the pooling and stacking layers, making this class reusable
        by sub-classes.

        :param basename: The base directory to load from
        :param kwargs: See below

        :Keyword Arguments:
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created

        :return: A restored model
        """
        _state = read_json("{}.state".format(basename))
        if __version__ != _state['version']:
            logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        _state['sess'] = kwargs.pop('sess', create_session())
        with _state['sess'].graph.as_default():
            embeddings_info = _state.pop('embeddings')
            embeddings = reload_embeddings(embeddings_info, basename)
            # If there is a kwarg that is the same name as an embedding object that
            # is taken to be the input of that layer. This allows for passing in
            # subgraphs like from a tf.split (for data parallel) or preprocessing
            # graphs that convert text to indices
            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            # TODO: convert labels into just another vocab and pass number of labels to models.
            labels = read_json("{}.labels".format(basename))
            model = cls.create(embeddings, labels, **_state)
            model._state = _state
            if kwargs.get('init', True):
                model.sess.run(tf.global_variables_initializer())
            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
            return model

    @property
    def lengths_key(self):
        return self._lengths_key

    @lengths_key.setter
    def lengths_key(self, value):
        self._lengths_key = value

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        """The main method for creating all :class:`ClassifierBasedModel` types.

        This method instantiates a model with pooling and optional stacking layers.
        Many of the arguments provided are reused by each implementation, but some sub-classes need more
        information in order to properly initialize.  For this reason, the full list of keyword args are passed
        to the :method:`pool` and :method:`stacked` methods.

        :param embeddings: This is a dictionary of embeddings, mapped to their numerical indices in the lookup table
        :param labels: This is a list of the `str` labels
        :param kwargs: There are sub-graph specific Keyword Args allowed for e.g. embeddings. See below for known args:

        :Keyword Arguments:
        * *gpus* -- (``int``) How many GPUs to split training across.  If called this function delegates to
            another class `ClassifyParallelModel` which creates a parent graph and splits its inputs across each
            sub-model, by calling back into this exact method (w/o this argument), once per GPU
        * *model_type* -- The string name for the model (defaults to `default`)
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created
        * *lengths_key* -- (``str``) Specifies which `batch_dict` property should be used to determine the temporal length
            if this is not set, it defaults to either `word`, or `x` if `word` is also not a feature
        * *finetune* -- Are we doing fine-tuning of word embeddings (defaults to `True`)
        * *mxlen* -- The maximum signal (`x` tensor temporal) length (defaults to `100`)
        * *dropout* -- This indicates how much dropout should be applied to the model when training.
        * *filtsz* -- This is actually a top-level param due to an unfortunate coupling between the pooling layer
            and the input, which, for convolution, requires input padding.

        :return: A fully-initialized tensorflow classifier
        """
        TRAIN_FLAG()

        model = cls()
        model.embeddings = {}
        for k, embedding in embeddings.items():
            model.embeddings[k] = embedding.detached_ref()

        model.lengths_key = kwargs.get('lengths_key')
        inputs = {}
        if model.lengths_key is not None:
            model._unserializable.append(model.lengths_key)
            model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
            inputs['lengths'] = model.lengths
        else:
            model.lengths = None

        model._record_state(**kwargs)
        for k, embedding in model.embeddings.items():
            x = kwargs.get(k, embedding.create_placeholder(name=k))
            inputs[k] = x


        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.sess = kwargs.get('sess', create_session())
        model.labels = labels

        nc = len(model.labels)
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, nc], name="y"))
        model.create_layers(**kwargs)
        model.logits = tf.identity(model.layers(inputs), name="logits")
        model.best = tf.argmax(model.logits, 1, name="best")
        model.probs = tf.nn.softmax(model.logits, name="probs")
        return model

    def create_layers(self, **kwargs):
        pass

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        return EmbeddingsStack(self.embeddings)


class EmbedPoolStackClassifier(ClassifierModelBase):

    def __init__(self):
        super(EmbedPoolStackClassifier, self).__init__()

    def create_layers(self, **kwargs):
        embeddings_stack = self.embed(**kwargs)

        nc = len(self.labels)
        pool = self.pool(embeddings_stack.dsz, **kwargs)
        stacking = self.stacked(**kwargs)
        self.layers = EmbedPoolStackModel(nc, embeddings_stack, pool, stacking)


    def pool(self, dsz, **kwargs):
        """This method performs a transformation between a temporal signal and a fixed representation

        :param word_embeddings: The output of the embedded lookup, which is the starting point for this operation
            :param dsz: The depth of the embeddings
        :param init: The tensorflow initializer to use for these methods
            :param kwargs: Model-specific arguments
        :return: A fixed representation of the data"""

    def stacked(self, **kwargs):
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
            return None
        return DenseStack(None, hszs, pdrop_value=self.pdrop_value)


@register_model(task='classify', name='default')
class ConvModel(EmbedPoolStackClassifier):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width

    """

    def __init__(self):
        """Constructor
        """
        super(ConvModel, self).__init__()

    def pool(self, dsz, **kwargs):
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling

        :param dsz: Embedding output size
        :param kwargs: See below

        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use

        :return:
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        return WithDropout(ParallelConv(dsz, cmotsz, filtsz), self.pdrop_value)


@register_model(task='classify', name='lstm')
class LSTMModel(EmbedPoolStackClassifier):
    """A simple single-directional single-layer LSTM. No layer-stacking.

    """

    def __init__(self):
        super(LSTMModel, self).__init__()
        self._vdrop = None

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def pool(self, dsz, **kwargs):
        """LSTM with dropout yielding a final-state as output

        :param word_embeddings: The input word embeddings
        :param dsz: The input word embedding depth
        :param kwargs: See below

        :Keyword Arguments:
        * *rnnsz* -- (``int``) The number of hidden units (defaults to `hsz`)
        * *hsz* -- (``int``) backoff for `rnnsz`, typically a result of stacking params.  This keeps things simple so
          its easy to do things like residual connections between LSTM and post-LSTM stacking layers

        :return:
        """
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        vdrop = bool(kwargs.get('variational', False))
        if type(hsz) is list:
            hsz = hsz[0]

        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        nlayers = int(kwargs.get('layers', 1))

        if rnntype == 'blstm':
            return BiLSTMEncoder(None, hsz, nlayers, self.pdrop_value, vdrop, output_fn=rnn_bi_hidden)
        return LSTMEncoder(None, hsz, nlayers, self.pdrop_value, vdrop, output_fn=rnn_hidden)


class NBowBase(EmbedPoolStackClassifier):

    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """
    def __init__(self):
        super(NBowBase, self).__init__()

    def stacked(self, **kwargs):
        """Force at least one hidden layer here

        :param kwargs:
        :return:
        """
        return super(NBowBase, self).stacked(**kwargs)


@register_model(task='classify', name='nbow')
class NBowModel(NBowBase):

    """Neural Bag-of-Words average pooling (standard) model"""
    def __init__(self):
        super(NBowModel, self).__init__()

    def pool(self, dsz, **kwargs):
        """Do average pooling on input embeddings, yielding a `dsz` output layer

        :param word_embeddings: The word embedding input
        :param dsz: The word embedding depth
        :param init: The tensorflow initializer
        :param kwargs: None
        :return: The average pooling representation
        """
        return tf.keras.layers.GlobalAveragePooling1D()


@register_model(task='classify', name='nbowmax')
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
        return tf.keras.layers.GlobalMaxPooling1D()


@register_model(task='classify', name='fine-tune')
class FineTuneModelClassifier(ClassifierModelBase):

    """Fine-tune based on pre-pooled representations"""
    def __init__(self):
        super(FineTuneModelClassifier, self).__init__()

    def create_layers(self, **kwargs):
        embeddings_stack = self.embed(**kwargs)
        nc = len(self.labels)
        self.layers = FineTuneModel(nc, embeddings_stack)


@register_model(task='classify', name='composite')
class CompositePoolingModel(EmbedPoolStackClassifier):

    """Fulfills pooling contract by aggregating pooling from a set of sub-models and concatenates each
    """
    def __init__(self):
        """
        Construct a composite pooling model
        """
        super(CompositePoolingModel, self).__init__()

    def pool(self, dsz, **kwargs):
        """Cycle each sub-model and call its pool method, then concatenate along final dimension

        :param word_embeddings: The input graph
        :param dsz: The number of input units
        :param init: The initializer operation
        :param kwargs:
        :return: A pooled composite output
        """
        SubModels = [eval(model) for model in kwargs.get('sub')]
        pooling = []
        for SubClass in SubModels:
            pooling.append(SubClass.pool(self, dsz, **kwargs))
        return tf.concat(pooling, -1)


