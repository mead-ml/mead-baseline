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


class TaggerModelBase(TaggerModel):
    """Tagger model base class for TensorFlow
    This class provides the implementation of the TaggerModel for TensorFlow
    """

    def __init__(self):
        """Create a tagger, nothing marked as unserializable
        """
        super(TaggerModelBase, self).__init__()
        self._unserializable = []
        self._lengths_key = None
        self._dropin_value = None
        self.saver = None


    def __init__(self):
        """Create a tagger, nothing marked as unserializable
        """
        super(TaggerModelBase, self).__init__()
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
        """Save the raw session tensors

        :param basename: The name of the model prefix
        :return: None
        """
        self.saver.save(self.sess, basename)

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

    def _record_state(self, **kwargs):
        """
        First, write out the embedding names, so we can recover those.  Then do a deepcopy on the model init params
        so that it can be recreated later.  Anything that is a placeholder directly on this model needs to be removed

        :param kwargs:
        :return:
        """
        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(
            self._unserializable,
            MAGIC_VARS,
            self.embeddings.keys(),
            (f"{k}_lengths" for k in self.embeddings.keys())
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

    def make_input(self, batch_dict, train=False):
        """Map a batch to a `feed_dict`.  Only used when `tf.dataset` is not being used

        When running a graph with placeholder inputs, this method provides an abstraction,
        converting the dictionary in the batch to what the model expects to fill its placeholders
        for a single batch.
        When `tf.dataset`s are in use, the features are connected directly to the graph,
        so there is no point in this method.

        :param batch_dict: A `Dict[str, tensor]` containing inputs to placeholders needed to run
        :param train: (`bool`) Is training on?
        :return:
        """
        feed_dict = new_placeholder_dict(train)
        for k in self.embeddings.keys():
            feed_dict["{}:0".format(k)] = self.drop_inputs(k, batch_dict[k], train)
        y = batch_dict.get('y', None)

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
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
        if __version__ != _state['version']:
            logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        _state['sess'] = kwargs.pop('sess', create_session())
        embeddings_info = _state.pop("embeddings")

        with _state['sess'].graph.as_default():
            embeddings = reload_embeddings(embeddings_info, basename)
            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            labels = read_json("{}.labels".format(basename))
            if _state.get('constraint_mask') is not None:
                # Dummy constraint values that will be filled in by the check pointing
                _state['constraint_mask'] = [tf.zeros((len(labels), len(labels))) for _ in range(2)]
            model = cls.create(embeddings, labels, **_state)
            model._state = _state
            model.create_loss()
            if kwargs.get('init', True):
                model.sess.run(tf.global_variables_initializer())
            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
            return model

    def save_using(self, saver):
        """Method to wire up the `tf.Saver`

        :param saver: The `tf.Saver`
        :return: None
        """
        self.saver = saver

    def create_loss(self):
        """Create the loss function and return it

        :return: The loss function
        """
        return self.layers.neg_log_loss(self.probs, self.y, self.lengths)

    def get_labels(self):
        """Get the labels (names of each class)

        :return: (`List[str]`) The labels
        """
        return self.labels

    def predict(self, batch_dict):
        """Take in a batch of data, and predict the tags

        :param batch_dict: A `Dict[str, tensor]` that is to be predicted
        :return: A batch-sized list of predictions
        """
        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict[self.lengths_key]
        bestv = self.sess.run(self.best, feed_dict=feed_dict)
        return [pij[:sl] for pij, sl in zip(bestv, lengths)]

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings
        :param kwargs:

        :return: A layer representing the embeddings
        """
        return EmbeddingsStack(self.embeddings, self.pdrop_value)

    def encode(self, **kwargs):
        """Provide a layer object that represents the `encode` phase of the model
       :param kwargs:
       :return: The encoder
       """
        pass

    def decode(self, **kwargs):
        """Provide a layer object that represents the `decode` phase of the model
        This will typically produce a CRF layer, or a greedy decoder

        :param kwargs:
        :return: Some decoder for the model
        """
        self.crf = bool(kwargs.get('crf', False))
        self.crf_mask = bool(kwargs.get('crf_mask', False))
        self.constraint_mask = kwargs.get('constraint_mask')
        if self.crf:
            return CRF(len(self.labels), self.constraint_mask)
        return TaggerGreedyDecoder(len(self.labels), self.constraint_mask)

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        """Create the model
        :param embeddings: A `dict` of input embeddings
        :param labels: The output labels for sequence tagging
        :param kwargs: See below
        :Keyword Arguments:

        * *lengths_key* (`str`) -- What is the name of the key that is used to get the full temporal length
        * *dropout* (`float`) -- The probability of dropout
        * *dropin* (`Dict[str, float]`) -- For each feature, how much input to dropout
        * *sess* -- An optional `tf.Session`, if not provided, this will be created
        * *span_type* -- (`str`) The type of input span
        * *username* -- (`str`) A username, defaults to the name of the user on this machine
        * *label* -- (`str`) An optional, human-readable label name.  Defaults to sha1 of this configuration
        * *variational* -- (`bool`) Should we do variational dropout
        * *rnntype* -- (`str`) -- The type of RNN (if this is an RNN), defaults to 'blstm'
        * *layers* -- (`int`) -- The number of layers to apply on the encoder
        * *hsz* -- (`int`) -- The number of hidden units for the encoder

        :return:
        """
        model = cls()
        model.embeddings = embeddings

        model.lengths_key = kwargs.get('lengths_key')
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model._unserializable.append(model.lengths_key)
        model._record_state(**kwargs)

        model.labels = labels
        nc = len(labels)

        # This only exists to make exporting easier
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.dropin_value = kwargs.get('dropin', {})
        model.sess = kwargs.get('sess', create_session())

        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        model.pdrop_in = kwargs.get('dropin', 0.0)
        model.labels = labels
        model.span_type = kwargs.get('span_type')

        inputs = {'lengths': model.lengths}
        for k, embedding in embeddings.items():
            x = kwargs.get(k, embedding.create_placeholder(name=k))
            inputs[k] = x

        embed_model = model.embed(**kwargs)
        transduce_model = model.encode(**kwargs)
        decode_model = model.decode(**kwargs)

        model.layers = TagSequenceModel(nc, embed_model, transduce_model, decode_model)
        model.probs = model.layers.transduce(inputs)
        model.best = model.layers.decode(model.probs, model.lengths)
        return model


@register_model(task='tagger', name='default')
class RNNTaggerModel(TaggerModelBase):

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    def encode(self, **kwargs):
        self.vdrop = kwargs.get('variational', False)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])

        Encoder = BiLSTMEncoder if rnntype == 'blstm' else LSTMEncoder
        return Encoder(hsz, nlayers, self.pdrop_value, self.vdrop, rnn_signal)


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(TaggerModelBase):

    def __init__(self):
        super(CNNTaggerModel, self).__init__()

    def encode(self, **kwargs):
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])
        filts = kwargs.get('wfiltsz', None)
        if filts is None:
            filts = 5

        cnnout = ConvEncoderStack(hsz, filts, self.pdrop_value, nlayers)
        return cnnout
