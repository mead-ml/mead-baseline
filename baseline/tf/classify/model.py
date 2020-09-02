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
    MEAD_HUB_MODULES
)
from baseline.model import ClassifierModel, register_model
from baseline.tf.tfy import (
    TRAIN_FLAG,
    reload_embeddings,
    new_placeholder_dict,
    tf_device_wrapper,
    create_session,
    BaseLayer,
    TensorDef
)


logger = logging.getLogger('baseline')


class ClassifierModelBase(tf.keras.Model, ClassifierModel):
    """Base for all baseline implementations of token-based classifiers

    This class provides a loose skeleton around which the baseline models
    are built.  It is built on the Keras Model base, and fulfills the `ClassifierModel` interface.
    To override this class, the use would typically override the `create_layers` function which will
    create and attach all sub-layers of this model into the class, and the `call` function which will
    give the model some implementation to call on forward.
    """
    def __init__(self, name=None):
        """Base
        """
        super().__init__(name=name)
        self._unserializable = []

    def set_saver(self, saver):
        self.saver = saver

    def save_values(self, basename):
        """Save tensor files out

        :param basename: Base name of model
        :return:
        """
        if not tf.executing_eagerly():
            self.saver.save(self.sess, basename, write_meta_graph=False)
        else:
            self.save_weights(f"{basename}.wgt")

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """
        write_json(self._state, '{}.state'.format(basename))
        write_json(self.labels, '{}.labels'.format(basename))
        for key, embedding in self.embeddings.items():
            if hasattr(embedding, 'save_md'):
                embedding.save_md('{}-{}-md.json'.format(basename, key))

    def _record_state(self, embeddings: Dict[str, BaseLayer], **kwargs):
        embeddings_info = {}
        for k, v in embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(
            self._unserializable,
            MAGIC_VARS,
            embeddings.keys(),
            (f'{k}_lengths' for k in embeddings.keys())
        ))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
            'hub_modules': MEAD_HUB_MODULES
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

    def predict_batch(self, batch_dict):

        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output. You can use this method directly if you have vector input, or you can use the `ClassifierService`
        which can convert directly from text durign its `transform`.  That method calls this one underneath.

        :param batch_dict: (``dict``) Contains any inputs to embeddings for this model
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """

        batch_dict = self.make_input(batch_dict)
        if not tf.executing_eagerly():
            probs = self.sess.run(self.probs, batch_dict)
        else:
            probs = tf.nn.softmax(self(batch_dict)).numpy()
        return probs

    def predict(self, batch_dict, raw=False, dense=False):

        """This method provides a basic routine to run "inference" or predict outputs based on data.
        It runs the `x` tensor in (`BxT`), and turns dropout off, running the network all the way to a softmax
        output. You can use this method directly if you have vector input, or you can use the `ClassifierService`
        which can convert directly from text durign its `transform`.  That method calls this one underneath.

        :param batch_dict: (``dict``) Contains any inputs to embeddings for this model
        :return: Each outcome as a ``list`` of tuples `(label, probability)`
        """

        probs = self.predict_batch(batch_dict)
        if raw and not dense:
            logger.warning("Warning: `raw` parameter is deprecated pass `dense=True` to get back values as a single tensor")
            dense = True
        if dense:
            return probs
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
        if not tf.executing_eagerly():
            batch_for_model = new_placeholder_dict(train)

            for k in self.embeddings.keys():
                batch_for_model["{}:0".format(k)] = batch_dict[k]

            # Allow us to track a length, which is needed for BLSTMs
            if self.lengths_key is not None:
                batch_for_model[self.lengths] = batch_dict[self.lengths_key]

            if y is not None:
                batch_for_model[self.y] = fill_y(len(self.labels), y)

        else:
            SET_TRAIN_FLAG(train)
            batch_for_model = {}
            for k in self.embeddings.keys():
                batch_for_model[k] = batch_dict[k]

            # Allow us to track a length, which is needed for BLSTMs
            if self.lengths_key is not None:
                batch_for_model["lengths"] = batch_dict[self.lengths_key]

        return batch_for_model

    def get_labels(self) -> List[str]:
        """Get the string labels back

        :return: labels
        """
        return self.labels

    @classmethod
    @tf_device_wrapper
    def load(cls, basename: str, **kwargs) -> 'ClassifierModelBase':
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

        if not tf.executing_eagerly():
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
                labels = read_json("{}.labels".format(basename))
                model = cls.create(embeddings, labels, **_state)
                model._state = _state
                if kwargs.get('init', True):
                    model.sess.run(tf.compat.v1.global_variables_initializer())
                model.saver = tf.compat.v1.train.Saver()
                model.saver.restore(model.sess, basename)
        else:
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
            model.load_weights(f"{basename}.wgt")
        return model

    @property
    def lengths_key(self) -> str:
        return self._lengths_key

    @lengths_key.setter
    def lengths_key(self, value: str):
        self._lengths_key = value

    @classmethod
    def create(cls, embeddings: Dict[str, BaseLayer], labels: List[str], **kwargs) -> 'ClassifierModelBase':
        """The main method for creating all :class:`ClassifierBasedModel` types.

        This method typically instantiates a model with pooling and optional stacking layers.
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
        model = cls(name=kwargs.get('name'))
        #embeddings_ = {}
        #for k, embedding in embeddings.items():
        #    embeddings_[k] = embedding #.detached_ref()

        model.lengths_key = kwargs.get('lengths_key')

        if not tf.executing_eagerly():

            inputs = {}
            if model.lengths_key is not None:
                model._unserializable.append(model.lengths_key)
                model.lengths = kwargs.get('lengths', tf.compat.v1.placeholder(tf.int32, [None], name="lengths"))
                inputs['lengths'] = model.lengths
            else:
                model.lengths = None

        model._record_state(embeddings, **kwargs)

        nc = len(labels)
        if not tf.executing_eagerly():
            model.y = kwargs.get('y', tf.compat.v1.placeholder(tf.int32, [None, nc], name="y"))
            for k, embedding in embeddings.items():
                x = kwargs.get(k, embedding.create_placeholder(name=k))
                inputs[k] = x

            model.sess = kwargs.get('sess', create_session())

        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.labels = labels
        model.create_layers(embeddings, **kwargs)

        if not tf.executing_eagerly():
            model.logits = tf.identity(model(inputs), name="logits")
            model.best = tf.argmax(model.logits, 1, name="best")
            model.probs = tf.nn.softmax(model.logits, name="probs")
        return model

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """


class EmbedPoolStackClassifier(ClassifierModelBase):
    """Provides a simple but effective base for most `ClassifierModel`s

    This class provides a common base for classifiers by identifying and codifying
    and idiomatic pattern where a typical classifier may be though of as a composition
    between a stack of embeddings, followed by a pooling operation yielding a fixed length
    tensor, followed by one or more dense layers, and ultimately, a projection to the output space.

    To provide an useful interface to sub-classes, we override the `create_layers` to provide a hook
    for each layer identified in this idiom, and leave the implementations up to the sub-class.

    We also fully implement the `call` method.

    """
    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        self.stack_model = self.init_stacked(self.pool_model.output_dim, **kwargs)
        self.output_layer = self.init_output(**kwargs)

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings
        :param kwargs: See below

        :Keyword Arguments:
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_name* (``str``) Optional override to Keras default names
        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        name = kwargs.get('embeddings_name')
        return EmbeddingsStack(embeddings, self.pdrop_value, reduction=reduction, name=name)

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a pooling operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A pooling operation
        """

    def init_stacked(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs: See below

        :keyword arguments:

        * *hsz* (``list``), defaults to nothing, in which case this function is pass-through
        * *stacked_name* (``str``) Optional override to stacking name

        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        name = kwargs.get('stacked_name')
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop_value, name=name)

    def init_output(self, **kwargs):
        """Provide a projection from the encoder output to the number of labels

        This projection typically will not include any activation, since its output is the logits that
        the decoder is built on

        :param kwargs: See below

        :keyword arguments:
        * *output_name* (``str``) Optional override to default Keras layer name
        :return: A projection from the encoder output size to the final number of labels
        """
        name = kwargs.get('output_name')
        return tf.keras.layers.Dense(len(self.labels), name=name)

    def call(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """
        lengths = inputs.get("lengths")
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        stacked = self.stack_model(pooled)
        return self.output_layer(stacked)


@register_model(task='classify', name='default')
class ConvModel(EmbedPoolStackClassifier):
    """Current default model for `baseline` classification.  Parallel convolutions of varying receptive field width
    """

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Do parallel convolutional filtering with varied receptive field widths, followed by max-over-time pooling

        :param input_dim: Embedding output size
        :param kwargs: See below

        :Keyword Arguments:
        * *cmotsz* -- (``int``) The number of convolutional feature maps for each filter
            These are MOT-filtered, leaving this # of units per parallel filter
        * *filtsz* -- (``list``) This is a list of filter widths to use
        * *pool_name* -- (``str``) Optional name to override default Keras layer name
        :return: A pooling layer
        """
        cmotsz = kwargs['cmotsz']
        filtsz = kwargs['filtsz']
        name = kwargs.get('pool_name')
        return WithoutLength(WithDropout(ParallelConv(input_dim, cmotsz, filtsz, name=name), self.pdrop_value))


@register_model(task='classify', name='lstm')
class LSTMModel(EmbedPoolStackClassifier):
    """A simple single-directional single-layer LSTM. No layer-stacking.
    """

    def __init__(self, name=None):
        super().__init__(name=name)
        self._vdrop = None

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """LSTM with dropout yielding a final-state as output

        :param input_dim: The input word embedding depth
        :param kwargs: See below

        :Keyword Arguments:
        * *rnnsz* -- (``int``) The number of hidden units (defaults to `hsz`)
        * *rnntype/rnn_type* -- (``str``) The RNN type, defaults to `lstm`, other valid values: `blstm`
        * *hsz* -- (``int``) backoff for `rnnsz`, typically a result of stacking params.  This keeps things simple so
          its easy to do things like residual connections between LSTM and post-LSTM stacking layers
        * *pool_name* -- (``str``) Optional name to override default Keras layer name

        :return: A pooling layer
        """
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 100))
        vdrop = bool(kwargs.get('variational', False))
        if type(hsz) is list:
            hsz = hsz[0]

        rnntype = kwargs.get('rnn_type', kwargs.get('rnntype', 'lstm'))
        nlayers = int(kwargs.get('layers', 1))
        name = kwargs.get('pool_name')
        if rnntype == 'blstm':
            return BiLSTMEncoderHidden(None, hsz, nlayers, self.pdrop_value, vdrop, name=name)
        return LSTMEncoderHidden(None, hsz, nlayers, self.pdrop_value, vdrop, name=name)


class NBowModelBase(EmbedPoolStackClassifier):
    """Neural Bag-of-Words Model base class.  Defines stacking of fully-connected layers, but leaves pooling to derived
    """

    def init_stacked(self, **kwargs):
        """Produce a stacking operation that will be used in the model, defaulting to a single layer

        :param input_dim: The input dimension size
        :param kwargs: See below

        :Keyword Arguments:
        * *hsz* -- (``List[int]``) The number of hidden units (defaults to 100)
        * *stacked_name* -- (``str``) Optional name to override default Keras layer name
        """
        kwargs.setdefault('hsz', [100])
        return super().stacked(**kwargs)


@register_model(task='classify', name='nbow')
class NBowModel(NBowModelBase):
    """Neural Bag-of-Words average pooling (standard) model"""

    def init_pool(self, input_dim: int, **kwargs):
        """Do average pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: See below

        :keyword arguments:

        * *pool_name* -- (``str``) Optional name to override default Keras layer name


        :return: The average pooling representation
        """
        name = kwargs.get('pool_name')
        return MeanPool1D(input_dim, name=name)


@register_model(task='classify', name='nbowmax')
class NBowMaxModel(NBowModelBase):
    """Max-pooling model for Neural Bag-of-Words.  Sometimes does better than avg pooling
    """

    def init_pool(self, input_dim: int, **kwargs) -> BaseLayer:
        """Do max pooling on input embeddings, yielding a `dsz` output layer

        :param input_dim: The word embedding depth
        :param kwargs: See below

        :keyword arguments:
        * *pool_name* -- (``str``) Optional name to override default Keras layer name

        :return: The max pooling representation
        """
        name = kwargs.get('pool_name')
        return WithoutLength(tf.keras.layers.GlobalMaxPooling1D(name=name))


@register_model(task='classify', name='fine-tune')
class FineTuneModelClassifier(ClassifierModelBase):
    """Fine-tune based on pre-pooled representations"""

    def init_embed(self, embeddings: Dict[str, TensorDef], **kwargs) -> BaseLayer:

        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings
        :param kwargs: See below

        :Keyword Arguments:
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings
        * *embeddings_name* (``str``) Optional override to Keras default names
        * *embeddings_dropout* (``float``) how much dropout post-reduction (defaults to 0.0)
        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        reduction = kwargs.get('embeddings_reduction', 'concat')
        embeddings_dropout = float(kwargs.get('embeddings_dropout', 0.0))
        name = kwargs.get('embeddings_name')
        return EmbeddingsStack(embeddings, embeddings_dropout, reduction=reduction, name=name)

    def init_stacked(self, input_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs: See below

        :keyword arguments:

        * *hsz* (``list``), defaults to nothing, in which case this function is pass-through
        * *stacked_name* (``str``) Optional override to stacking name

        :return: A stacking operation (or None)
        """
        hszs = listify(kwargs.get('hsz', []))
        if not hszs:
            return PassThru(input_dim)
        name = kwargs.get('stacked_name')
        return DenseStack(input_dim, hszs, pdrop_value=self.pdrop_value, name=name)

    def init_output(self, **kwargs):
        """Provide a projection from the encoder output to the number of labels

        This projection typically will not include any activation, since its output is the logits that
        the decoder is built on

        :param kwargs: See below

        :keyword arguments:
        * *output_name* (``str``) Optional override to default Keras layer name
        :return: A projection from the encoder output size to the final number of labels
        """
        name = kwargs.get('output_name')
        return tf.keras.layers.Dense(len(self.labels), name=name)

    def create_layers(self, embeddings, **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.stack_model = self.init_stacked(self.embeddings.output_dim, **kwargs)
        self.output_layer = self.init_output(**kwargs)

    def call(self, inputs):
        base_layers = self.embeddings(inputs)
        stacked = self.stack_model(base_layers)
        return self.output_layer(stacked)


@register_model(task='classify', name='fine-tune-paired')
class FineTunePairedClassifierModel(FineTuneModelClassifier):

    """Fine-tuning model for pairs

    This model encodes a pair as a single utterance using some encoding scheme defined in
    ``_convert_pair`` which is fed directly into the fine-tuning model.

    For BERT, this simply encodes the input key pair as a single utterance while building
    a token-type vector.

    For the input, we will assume that the vectorizer will be producing a start token and an end token.
    We will simply remove the start token from the second sentence and concatenate
    [CLS] this is sentence one [SEP]

    [CLS] this is sentence two [SEP]


    """
    def _convert_pair(self, key, batch_dict, example_dict):

        toks = batch_dict[key]
        token_type_key = f"{key}_tt"
        #eager = tf.executing_eagerly()
        target_key = key  # if eager else f"{key}:0"
        tt = batch_dict.get(token_type_key)
        if tt is not None:
            #if not eager:
            #    raise Exception("We arent currently supporting non-eager mode with token_types")
            #else:
            example_dict[target_key] = (toks, tt)
        else:
            example_dict[target_key] = toks

    def call(self, inputs):
        inputs_for_model = {}
        for key in self.embeddings.keys():
            self._convert_pair(key, inputs, inputs_for_model)
        base_layers = self.embeddings(inputs_for_model)
        stacked = self.stack_model(base_layers)
        return self.output_layer(stacked)

    #def make_input(self, batch_dict, train=False):
    #    """Transform a `batch_dict` into a TensorFlow `feed_dict`
    #
    #    :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
    #    :param train: (``bool``) Are we training.  Defaults to False
    #    :return:
    #    """
    #    y = batch_dict.get('y', None)
    #    if not tf.executing_eagerly():
    #        batch_for_model = new_placeholder_dict(train)
    #
    #        for key in self.embeddings.keys():
    #            self._convert_pair(key, batch_dict, batch_for_model)
    #
    #        if y is not None:
    #            batch_for_model[self.y] = fill_y(len(self.labels), y)
    #
    #    else:
    #        SET_TRAIN_FLAG(train)
    #        batch_for_model = {}
    #        for key in self.embeddings.keys():
    #            self._convert_pair(key, batch_dict, batch_for_model)
    #    return batch_for_model


@register_model(task='classify', name='composite')
class CompositePoolingModel(EmbedPoolStackClassifier):
    """Fulfills pooling contract by aggregating pooling from a set of sub-models and concatenates each
    """

    def pool(self, dsz, **kwargs):
        """Cycle each sub-model and call its pool method, then concatenate along final dimension

        :param word_embeddings: The input graph
        :param dsz: The number of input units
        :param init: The initializer operation
        :param kwargs:
        :return: A pooled composite output
        """
        SubModels = [eval(model) for model in kwargs.get('sub')]
        models = []
        for SubClass in SubModels:
            models.append(SubClass.pool(self, dsz, **kwargs))
        return CompositeModel(models)
