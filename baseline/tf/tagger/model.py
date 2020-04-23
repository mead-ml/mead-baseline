"""Provides tagger models for TensorFlow
"""
from baseline.model import TaggerModel
from itertools import chain
from baseline.tf.tfy import *
from eight_mile.tf.layers import *
from baseline.utils import ls_props, read_json, write_json, MAGIC_VARS
from baseline.tf.embeddings import *
from baseline.version import __version__
from baseline.model import register_model
from baseline.utils import listify, Offsets

logger = logging.getLogger('baseline')


class TaggerModelBase(tf.keras.Model, TaggerModel):
    """Base class for tagger models

    This class provides the model base for tagging.  To create a tagger, overload `create_layers()` and `forward()`.
    Most implementations should be able to subclass the `AbstractEncoderTaggerModel`, which inherits from this and imposes
    additional structure
    """
    def __init__(self):
        """Create a tagger, nothing marked as unserializable
        """
        super().__init__()
        self._unserializable = []
        self._lengths_key = None
        self._dropin_value = None
        self.saver = None

    @property
    def lengths_key(self):
        """Property for the name of the field that is used for lengths keying

        :return: (`str`) The name of the field
        """
        return self._lengths_key

    @lengths_key.setter
    def lengths_key(self, value):
        """Set the lengths key

        :param value: (`str`) The value to set the lengths key to
        :return: None
        """
        self._lengths_key = value

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
        """
        This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename: The name of the model prefix
        :return: None
        """
        write_json(self._state, '{}.state'.format(basename))
        write_json(self.labels, '{}.labels'.format(basename))
        for key, embedding in self.embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

    def _record_state(self, embeddings, **kwargs):
        """
        First, write out the embedding names, so we can recover those.  Then do a deepcopy on the model init params
        so that it can be recreated later.  Anything that is a placeholder directly on this model needs to be removed

        :param kwargs:
        :return:
        """
        embeddings_info = {}
        for k, v in embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(
            self._unserializable,
            MAGIC_VARS,
            embeddings.keys(),
            (f"{k}_lengths" for k in embeddings.keys())
        ))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
        })
        if 'constraint_mask' in kwargs:
            self._state['constraint_mask'] = True

    @property
    def dropin_value(self):
        """Dropout on the input as a `Dict[str, float]`, one per embedding (keyed off the feature name)

        :return: `Dict[str, float]` containing this information
        """
        return self._dropin_value

    @dropin_value.setter
    def dropin_value(self, dict_value):
        """Set dictionary for dropout on the input values

        :param dict_value: `Dict[str, float]` containing this information
        :return: None
        """
        self._dropin_value = dict_value

    def drop_inputs(self, key, x, do_dropout):
        """Do dropout on inputs, using the dropout value (or none if not set)
        This works by applying a dropout mask with the probability given by a
        value within the `dropin_value: Dict[str, float]`, keyed off the text name
        of the feature

        :param key: The feature name
        :param x: The tensor to drop inputs for
        :param do_dropout: A `bool` specifying if dropout is turned on
        :return: The dropped out tensor
        """
        v = self.dropin_value.get(key, 0)
        if do_dropout and v > 0.0:
            drop_indices = np.where((np.random.random(x.shape) < v) & (x != Offsets.PAD))
            x[drop_indices[0], drop_indices[1]] = Offsets.UNK
        return x

    def make_input(self, batch_dict: Dict[str, TensorDef], train: bool = False) -> Dict[str, TensorDef]:
        """Transform a `batch_dict` into format suitable for tagging

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :param train: (``bool``) Are we training.  Defaults to False
        :return: A dictionary representation of this batch suitable for processing
        """
        y = batch_dict.get('y', None)
        if not tf.executing_eagerly():
            batch_for_model = new_placeholder_dict(train)

            for k in self.embeddings.keys():
                batch_for_model["{}:0".format(k)] = self.drop_inputs(k, batch_dict[k], train)

            # Allow us to track a length, which is needed for BLSTMs
            batch_for_model[self.lengths] = batch_dict[self.lengths_key]

            if y is not None:
                batch_for_model[self.y] = y
        else:
            SET_TRAIN_FLAG(train)
            batch_for_model = {}
            for k in self.embeddings.keys():
                batch_for_model[k] = self.drop_inputs(k, batch_dict[k], train)

            # Allow us to track a length, which is needed for BLSTMs
            if self.lengths_key is not None:
                batch_for_model["lengths"] = batch_dict[self.lengths_key]

        return batch_for_model

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def save(self, basename: str):
        """Save a model, using the parameter as a prefix

        :param basename: The prefix for the model including the directories and the base of the name to use
        :return: None
        """
        self.save_md(basename)
        self.save_values(basename)

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
        # FIXME: Somehow not writing this anymore
        #if __version__ != _state['version']:
        #    logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        if not tf.executing_eagerly():

            _state['sess'] = kwargs.pop('sess', create_session())
            embeddings_info = _state.pop("embeddings")

            with _state['sess'].graph.as_default():
                embeddings = reload_embeddings(embeddings_info, basename)
                for k in embeddings_info:
                    if k in kwargs:
                        _state[k] = kwargs[k]
                labels = read_json("{}.labels".format(basename))
                # FIXME: referring to the `constraint_mask` in the base where its not mentioned isnt really clean
                if _state.get('constraint_mask') is not None:
                    # Dummy constraint values that will be filled in by the check pointing
                    _state['constraint_mask'] = [np.zeros((len(labels), len(labels))) for _ in range(2)]
                model = cls.create(embeddings, labels, **_state)
                model._state = _state
                model.create_loss()
                if kwargs.get('init', True):
                    model.sess.run(tf.compat.v1.global_variables_initializer())
                model.saver = tf.compat.v1.train.Saver()
                model.saver.restore(model.sess, basename)
        else:
            embeddings_info = _state.pop('embeddings')
            embeddings = reload_embeddings(embeddings_info, basename)

            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            # TODO: convert labels into just another vocab and pass number of labels to models.
            labels = read_json("{}.labels".format(basename))
            # FIXME: referring to the `constraint_mask` in the base where its not mentioned isnt really clean
            if _state.get('constraint_mask') is not None:
                # Dummy constraint values that will be filled in by the check pointing
                _state['constraint_mask'] = [np.zeros((len(labels), len(labels))) for _ in range(2)]
            model = cls.create(embeddings, labels, **_state)
            model._state = _state
            model.load_weights(f"{basename}.wgt")
        return model

    def save_using(self, saver):
        """Method to wire up the `tf.Saver`

        :param saver: The `tf.Saver`
        :return: None
        """
        self.saver = saver

    def get_labels(self) -> List[str]:
        """Get the labels (names of each class)

        :return: (`List[str]`) The labels
        """
        return self.labels

    def predict(self, batch_dict: Dict[str, TensorDef]) -> TensorDef:
        """Take in a batch of data, and predict the tags

        :param batch_dict: A `Dict[str, tensor]` that is to be predicted
        :return: A batch-sized list of predictions
        """
        lengths = batch_dict[self.lengths_key]
        batch_dict = self.make_input(batch_dict)
        if not tf.executing_eagerly():
            return self.sess.run(self.best, feed_dict=batch_dict)
        else:
            return self(batch_dict).numpy()

    def call(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Take the input and produce the best path of labels out

        :param inputs: The feature indices for the input
        :return: The most likely path through the output labels
        """

    @classmethod
    def create(cls, embeddings, labels, **kwargs) -> 'TaggerModelBase':
        """Create the model
        :param embeddings: A `dict` of input embeddings
        :param labels: The output labels for sequence tagging
        :param kwargs: See below
        :Keyword Arguments:

        * *lengths_key* (`str`) -- What is the name of the key that is used to get the full temporal length
        * *dropout* (`float`) -- The probability of dropout
        * *dropin* (`Dict[str, float]`) -- For each feature, how much input to dropout
        * *sess* -- An optional `tf.compat.v1.Session`, if not provided, this will be created
        * *span_type* -- (`str`) The type of input span
        * *username* -- (`str`) A username, defaults to the name of the user on this machine
        * *label* -- (`str`) An optional, human-readable label name.  Defaults to sha1 of this configuration
        * *variational* -- (`bool`) Should we do variational dropout
        * *rnntype* -- (`str`) -- The type of RNN (if this is an RNN), defaults to 'blstm'
        * *layers* -- (`int`) -- The number of layers to apply on the encoder
        * *hsz* -- (`int`) -- The number of hidden units for the encoder

        :return: A newly created tagger
        """
        model = cls()

        model.lengths_key = kwargs.get('lengths_key')

        if not tf.executing_eagerly():

            inputs = {}
            for k, embedding in embeddings.items():
                x = kwargs.get(k, embedding.create_placeholder(name=k))
                inputs[k] = x
            model._unserializable.append(model.lengths_key)
            model.lengths = kwargs.get('lengths', tf.compat.v1.placeholder(tf.int32, [None], name="lengths"))
            inputs['lengths'] = model.lengths
            model.y = kwargs.get('y', tf.compat.v1.placeholder(tf.int32, [None, None], name="y"))
            model.sess = kwargs.get('sess', create_session())

        model._record_state(embeddings, **kwargs)
        model.labels = labels
        # This only exists to make exporting easier
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.dropin_value = kwargs.get('dropin', {})

        model.labels = labels
        model.span_type = kwargs.get('span_type')

        model.create_layers(embeddings, **kwargs)
        if not tf.executing_eagerly():
            model.best = model(inputs)
        return model

    def create_loss(self):
        """Create the loss function and return it

        :return: The loss function
        """


class AbstractEncoderTaggerModel(TaggerModelBase):
    """Class defining a typical flow for taggers.  Most taggers should extend this class

    This class provides the model base for tagging by providing specific hooks for each phase.  There are
    4 basic steps identified in this class:

    1. embed
    2. encode (transduction)
    3. proj (projection to the final number of labels)
    4. decode

    There is an `init_* method for each of this phases, allowing you to
    define and return a custom layer.

    The actual forward method is defined as a combination of these 3 steps, which includes a
    projection from the encoder output to the number of labels.

    Decoding in taggers refers to the process of selecting the best path through the labels and is typically
    implemented either as a constrained greedy decoder or as a CRF layer
    """
    def __init__(self):
        super().__init__()

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.encoder = self.init_encode(**kwargs)
        self.proj_layer = self.init_proj(**kwargs)
        self.decoder = self.init_decode(**kwargs)

    def call(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """Take the input and produce the best path of labels out

        :param inputs: The feature indices for the input
        :return: The most likely path through the output labels
        """
        self.probs = self.transduce(inputs)
        return self.decode(self.probs, inputs.get("lengths"))

    def create_loss(self):
        """Create the loss function and return it

        :return: The loss function
        """
        return self.decoder.neg_log_loss(self.probs, self.y, self.lengths)

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

    def init_encode(self, **kwargs) -> BaseLayer:
        """Provide a layer object that represents the `encode` phase of the model
       :param kwargs:
       :return: The encoder
       """

    def init_proj(self, **kwargs) -> BaseLayer:
        """Provide a projection from the encoder output to the number of labels

        This projection typically will not include any activation, since its output is the logits that
        the decoder is built on

        :param kwargs: See below

        :keyword arguments:
        * *proj_name* (``str``) Optional override to default Keras layer name
        :return: A projection from the encoder output size to the final number of labels
        """
        name = kwargs.get('proj_name')
        return tf.keras.layers.Dense(len(self.labels), name=name)

    def init_decode(self, **kwargs) -> BaseLayer:
        """Provide a layer object that represents the `decode` phase of the model
        This will typically produce a CRF layer, or a greedy decoder

        :param kwargs: See below

        :keyword arguments:
        * *crf* (``bool``) Is it a CRF?
        * *constraint_mask* (``bool``) Is there a CRF mask?
        * *decode_name* (``str``) Optional TF graph name to use, defaults to Keras layer default
        :return: Some decoder for the model
        """
        self.crf = bool(kwargs.get('crf', False))
        self.constraint_mask = kwargs.get('constraint_mask')
        name = kwargs.get('decode_name')
        if self.crf:
            return CRF(len(self.labels), self.constraint_mask, name=name)
        return TaggerGreedyDecoder(len(self.labels), self.constraint_mask, name=name)

    def transduce(self, inputs: Dict[str, TensorDef]) -> TensorDef:
        """This operation performs embedding of the input, followed by encoding and projection to logits

        :param inputs: The feature indices to embed
        :return: Transduced (post-encoding) output
        """
        lengths = inputs["lengths"]
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        transduced = self.proj_layer(self.encoder(embedded))
        return transduced

    def decode(self, tensor: TensorDef, lengths: TensorDef) -> TensorDef:
        """Take in the transduced (encoded) input and decode it

        :param tensor: Transduced input
        :param lengths: Valid lengths of the transduced input
        :return: A best path through the output
        """
        path, self.path_scores = self.decoder((tensor, lengths))
        return path


@register_model(task='tagger', name='default')
class RNNTaggerModel(AbstractEncoderTaggerModel):
    """RNN-based tagger implementation: this is the default tagger for mead-baseline

    Overload the encoder, typically as a BiLSTM
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, **kwargs):
        """Override the base method to produce an RNN transducer

        :param kwargs: See below

        :Keyword Arguments:
        * *rnntype* (``str``) The type of RNN, defaults to `blstm`
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *variational* (``bool``) Variational dropout
        * *encode_name* (``str``) A Keras layer name to provide to the encoder, default to Keras layer name
        :return: An encoder
        """
        self.vdrop = kwargs.get('variational', False)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = int(kwargs.get('layers', 1))
        hsz = int(kwargs['hsz'])
        name = kwargs.get('encode_name')
        Encoder = BiLSTMEncoderSequence if rnntype == 'blstm' else LSTMEncoderSequence
        return Encoder(None, hsz, nlayers, self.pdrop_value, self.vdrop, name=name)

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value


@register_model(task='tagger', name='transformer')
class TransformerTaggerModel(AbstractEncoderTaggerModel):
    """Transformer-based tagger model

    Overload the encoder using a length-aware Transformer
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, **kwargs):
        """Override the base method to produce an RNN transducer

        :param kwargs: See below

        :Keyword Arguments:
        * *num_heads* (``int``) The number of heads for multi-headed attention
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *d_ff* (``int``) The feed-forward layer size
        * *rpr_k* (``list`` or ``int``) The relative attention sizes.  If its a list, one scalar per layer, if its
          a scalar, apply same size to each layer
        :return: An encoder
        """
        layers = int(kwargs.get('layers', 1))
        num_heads = int(kwargs.get('num_heads', 4))
        pdrop = float(kwargs.get('dropout', 0.5))
        scale = False
        hsz = int(kwargs['hsz'])
        rpr_k = kwargs.get('rpr_k', 100)
        d_ff = kwargs.get('d_ff')
        encoder = TransformerEncoderStackWithLengths(num_heads, hsz, pdrop, scale, layers, d_ff=d_ff, rpr_k=rpr_k)
        return encoder


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(AbstractEncoderTaggerModel):
    """Convolutional (AKA TDNN) tagger

    Overload the encoder using a conv layer

    """
    def __init__(self):
        super().__init__()

    def init_encode(self, **kwargs):
        """Override the base method to produce an RNN transducer

        :param kwargs: See below

        :Keyword Arguments:
        * *layers* (``int``) The number of layers to stack
        * *hsz* (``int``) The number of hidden units for each layer in the encoder
        * *dropout* (``float``) The dropout rate, defaults
        * *wfiltsz* (``int``) The 1D filter size for the convolution
        :return: An encoder
        """
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])
        filts = kwargs.get('wfiltsz', None)
        if filts is None:
            filts = 5

        cnnout = ConvEncoderStack(None, hsz, filts, self.pdrop_value, nlayers)
        return cnnout


@register_model(task='tagger', name='pass')
class PassThruTaggerModel(AbstractEncoderTaggerModel):
    """A Pass-thru implementation of the encoder

    When we fine-tune our taggers from things like BERT embeddings, we might want to just pass through our
    embedding result directly to the output decoder.  This model provides a mechanism for this by providing
    a simple identity layer
    """
    def __init__(self):
        super().__init__()

    def init_encode(self, **kwargs) -> BaseLayer:
        """Identity layer encoder

        :param kwargs: None
        :return: An encoder
        """
        input_dim = self.embeddings.output_dim
        return WithoutLength(PassThru(input_dim))
