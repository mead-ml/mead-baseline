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
from baseline.model import DependencyParserModel, register_model
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


class DependencyParserModelBase(tf.keras.Model, DependencyParserModel):
    """Base for all baseline implementations of token-based classifiers

    This class provides a loose skeleton around which the baseline models
    are built.  It is built on the Keras Model base, and fulfills the `DependencyParserModel` interface.
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

    def predict_batch(self, batch_dict):

        batch_dict = self.make_input(batch_dict)
        arcs, rels = self(batch_dict)
        arcs = tf.nn.softmax(arcs).numpy()
        rels = tf.nn.softmax(rels).numpy()
        return arcs, rels

    def predict(self, batch_dict):
        probs = self.predict_batch(batch_dict)
        return probs

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
        """Transform a `batch_dict` into a TensorFlow `feed_dict`

        :param batch_dict: (``dict``) A dictionary containing all inputs to the embeddings for this model
        :param train: (``bool``) Are we training.  Defaults to False
        :return:
        """
        SET_TRAIN_FLAG(train)
        batch_for_model = {}
        for k in self.embeddings.keys():
            batch_for_model[k] = self.drop_inputs(k, batch_dict[k], train)
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
    def load(cls, basename: str, **kwargs) -> 'DependencyParserModelBase':
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
    def create(cls, embeddings: Dict[str, BaseLayer], labels, **kwargs) -> 'DependencyParserModelBase':
        """The main method for creating all :class:`DependencyParserBaseModel` types.

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

        :return: A fully-initialized tensorflow dependency parser
        """
        model = cls(name=kwargs.get('name'))
        model.lengths_key = kwargs.get('lengths_key')
        model._record_state(embeddings, **kwargs)
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.labels = labels["labels"]
        model.punct = labels["labels"].get("punct", Offsets.PAD)
        model.create_layers(embeddings, **kwargs)
        return model

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        """This method defines the model itself, and must be overloaded by derived classes

        This function will update `self` with the layers required to execute the `call()` method

        :param embeddings: The input feature indices
        :param kwargs:
        :return:
        """

    def decode(self, example, **kwargs):
        heads_pred, labels_pred = self(example)
        heads_pred = heads_pred.numpy()
        labels_pred = labels_pred.numpy()
        # Just do a quick greedy decode, pick the argmax of the heads, and for that head, pick the
        # argmax of the label
        B = labels_pred.shape[0]
        T = labels_pred.shape[1]

        # If there is padding, rip it off to the max sequence length so the tensors are the same size
        greedy_heads_pred = np.argmax(heads_pred, -1).reshape(-1)
        greedy_labels_pred = labels_pred.reshape(B * T, T, -1)[
            np.arange(len(greedy_heads_pred)), greedy_heads_pred].reshape(B, T, -1)

        greedy_labels_pred = np.argmax(greedy_labels_pred, -1)
        greedy_heads_pred = greedy_heads_pred.reshape(B, T)
        return greedy_heads_pred, greedy_labels_pred


@register_model(task='deps', name='default')
class BiAffineDependencyParser(DependencyParserModelBase):

    def create_layers(self, embeddings: Dict[str, TensorDef], **kwargs):
        self.embeddings = self.init_embed(embeddings, **kwargs)
        self.pool_model = self.init_pool(self.embeddings.output_dim, **kwargs)
        output_dim_arcs = kwargs.get('hsz_arcs', 500)
        self.arc_h = self.init_proj(output_dim_arcs, **kwargs)
        self.arc_d = self.init_proj(output_dim_arcs, **kwargs)
        output_dim_rels = kwargs.get('hsz_rels', 100)
        self.rel_h = self.init_proj(output_dim_rels, **kwargs)
        self.rel_d = self.init_proj(output_dim_rels, **kwargs)
        self.arc_attn = self.init_biaffine(self.arc_h.output_dim, 1, True, False)
        self.rel_attn = self.init_biaffine(self.rel_h.output_dim, len(self.labels), True, True)
        self.primary_key = self.lengths_key.split('_')[0]


    def init_proj(self, output_dim: int, **kwargs) -> BaseLayer:
        """Produce a stacking operation that will be used in the model

        :param input_dim: The input dimension size
        :param kwargs:
        :return: A stacking operation (or None)
        """
        return WithDropout(tf.keras.layers.Dense(output_dim, activation=get_activation(kwargs.get('activation', 'leaky_relu'))), pdrop=self.pdrop_value)

    def init_biaffine(self, input_dim: int, output_dim: int, bias_x: bool, bias_y: bool):
        return BilinearAttention(input_dim, output_dim, bias_x, bias_y)

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
        hsz = kwargs.get('rnnsz', kwargs.get('hsz', 800))
        layers = kwargs.get('layers', 3)
        return BiLSTMEncoderSequence(input_dim, hsz, layers, self.pdrop_value) #, variational=True)

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

    def call(self, inputs: Dict[str, TensorDef]) -> Tuple[TensorDef, TensorDef]:
        """Forward execution of the model.  Sub-classes typically shouldnt need to override

        :param inputs: An input dictionary containing the features and the primary key length
        :return: A tensor
        """

        lengths = inputs.get("lengths")
        mask = tf.sequence_mask(lengths)
        embedded = self.embeddings(inputs)
        embedded = (embedded, lengths)
        pooled = self.pool_model(embedded)
        arcs_h = self.arc_h(pooled)
        arcs_d = self.arc_d(pooled)
        rels_h = self.rel_h(pooled)
        rels_d = self.rel_d(pooled)
        score_arcs = self.arc_attn(arcs_d, arcs_h, mask)
        mask = tf.expand_dims(mask, 1)
        score_rels = tf.transpose(self.rel_attn(rels_d, rels_h, mask), [0, 2, 3, 1])
        return score_arcs, score_rels

