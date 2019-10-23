"""Language model baselines in TensorFlow
"""
from baseline.tf.tfy import *
from baseline.version import __version__
from baseline.model import LanguageModel, register_model
from baseline.tf.embeddings import *
from baseline.tf.tfy import new_placeholder_dict, TRAIN_FLAG, lstm_cell_w_dropout
from baseline.utils import read_json, write_json, MAGIC_VARS


class LanguageModelBase(tf.keras.Model, LanguageModel):
    """Base for all baseline implementations of LMs

    This class provides a loose skeleton around which the baseline models
    are built.  This essentially consists of dividing up the network into a logical separation between "embedding",
    or composition of lookup tables to build a vector representation of a temporal input, "decoding",
    or the conversion of temporal data to a decoded representation, and "output" --
    a projection to output space and a softmax
    """
    def __init__(self):
        """Construct a base LM
        """
        super().__init__()
        self.saver = None
        self.hsz = None
        self.probs = None
        self._unserializable = []

    def save_values(self, basename):
        """Save tensor files out

        :param basename: Base name of model
        :return:
        """
        if get_version(tf) < 2:
            self.saver.save(self.sess, basename)
        else:
            self.save_weights(f"{basename}.wgt")

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """

        write_json(self._state, basename + '.state')
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))

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

        self._state = {k: v for k, v in kwargs.items() if k not in self._unserializable + MAGIC_VARS + list(self.embeddings.keys())}
        self._state.update({
            "version": __version__,
            "embeddings": embeddings_info
        })

    def set_saver(self, saver):
        """Connect a `tf.Saver` to the model

        :param saver: A saver
        :return: None
        """
        self.saver = saver

    def decode(self, inputs, **kwargs):
        """Base method for decoding

        :param inputs: The outputs of the embeddings
        :param kwargs:
        :return:
        """
        pass

    def save(self, basename):
        """Save the model

        :param basename: The model prefix
        :return:
        """
        self.save_md(basename)
        self.save_values(basename)

    def _create_loss(self, scope):
        with tf.compat.v1.variable_scope(scope):
            vsz = self.embeddings[self.tgt_key].vsz
            targets = tf.reshape(self.y, [-1])
            bt_x_v = tf.nn.log_softmax(tf.reshape(self.logits, [-1, vsz]), axis=-1)
            one_hots = tf.one_hot(targets, vsz)
            example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
            loss = tf.reduce_mean(example_loss)
            return loss

    def create_loss(self):
        """Create training loss operator

        :return: loss
        """
        return self._create_loss(scope='loss')

    def create_test_loss(self):
        """Create test loss operator

        :return: loss
        """
        return self._create_loss(scope='test_loss')

    def make_input(self, batch_dict, train=False):
        """When we are running with `DataFeed`s, need to transform to `feed_dict`s

        :param batch_dict: The batch for a step
        :param train: (`bool`) Are we training (or evaluating)?
        :return: A `feed_dict`
        """
        if get_version(tf) < 2:
            batch_dict_for_model = new_placeholder_dict(train)

            for key in self.src_keys:

                batch_dict_for_model["{}:0".format(key)] = batch_dict[key]

            y = batch_dict.get('y')
            if y is not None:
                batch_dict_for_model[self.y] = batch_dict['y']

        else:
            SET_TRAIN_FLAG(train)

            batch_dict_for_model = {}
            for key in self.src_keys:
                batch_dict_for_model[key] = batch_dict[key]

        return batch_dict_for_model

    def predict(self, batch_dict):
        """Do prediction from a `batch_dict`

        :param batch_dict: A step of data
        :return: The softmax output for this step
        """
        batch_dict = self.make_input(batch_dict)
        if get_version(tf) < 2:
            step_softmax = self.sess.run(self.probs, batch_dict)
        else:
            step_softmax = tf.nn.softmax(self.impl(batch_dict))

        return step_softmax

    def call(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.impl.trainable_variables

    @property
    def variables(self):
        return self.impl.variables

    @classmethod
    def create(cls, embeddings, **kwargs):
        """Create the language model

        :param embeddings: A set of embeddings used
        :param kwargs: see below

        :Keyword Arguments:

        * *tgt_key* (`str`) -- Which vocabulary is the destination vocabulary
          (for example, you might have character inputs, or character + word inputs.  The outputs need to be specified)
        * *sess* (`tf.Session`) -- Optionally, pass in a session (or one will be created)
        * *pdrop* (`float`) -- The dropout probability
        * *y* -- Optional target.  If this is not passed in, a placeholder gets created
        * *hsz* (`int`) -- Number of hidden units per layers
        * *unif* (`float`) -- set the weights initializer to small random uniform values

        :return: The created model
        """
        lm = cls()
        lm.embeddings = {k: embedding.detached_ref() for k, embedding in embeddings.items()}
        lm.src_keys = kwargs.get('src_keys', lm.embeddings.keys())
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')

        lm._unserializable.append(lm.tgt_key)
        lm._record_state(**kwargs)
        inputs = {}
        lm.batchsz = 0

        if get_version(tf) < 2:

            for k, embedding in embeddings.items():
                x = kwargs.get(k, embedding.create_placeholder(name=k))
                lm.batchsz = tf.shape(x)[0]
                inputs[k] = x

            lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
            lm.sess = kwargs.get('sess', tf.Session())
        lm.pdrop_value = kwargs.pop('pdrop', 0.5)
        lm.hsz = kwargs.pop('hsz', None)
        embeddings_layer = lm.embed(**kwargs)
        nc = embeddings[lm.tgt_key].vsz
        encoder_layer = lm.decode(inputs, **kwargs)
        lm.impl = LangSequenceModel(nc, embeddings_layer, encoder_layer)

        if get_version(tf) < 2:
            lm.logits, lm.final_state = lm(inputs)
            lm.probs = tf.nn.softmax(lm.logits, name="softmax")

        return lm

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        src_embeddings = {k: self.embeddings[k] for k in self.src_keys}
        embed_output = EmbeddingsStack(src_embeddings, self.pdrop_value)
        return embed_output

    @classmethod
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
        if get_version(tf) < 2:
            _state = read_json(basename + '.state')
            _state['sess'] = kwargs.pop('sess', tf.Session())
            _state['model_type'] = kwargs.get('model_type', 'default')
            embeddings = {}
            embeddings_dict = _state.pop("embeddings")

            for key, class_name in embeddings_dict.items():
                md = read_json('{}-{}-md.json'.format(basename, key))
                embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
                Constructor = eval(class_name)
                embeddings[key] = Constructor(key, **embed_args)

            model = cls.create(embeddings, **_state)
            model._state = _state

            do_init = kwargs.get('init', True)
            if do_init:
                init = tf.global_variables_initializer()
                model.sess.run(init)

            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
        else:
            _state = read_json(basename + '.state')
            _state['model_type'] = kwargs.get('model_type', 'default')
            embeddings = {}
            embeddings_dict = _state.pop("embeddings")

            for key, class_name in embeddings_dict.items():
                md = read_json('{}-{}-md.json'.format(basename, key))
                embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
                Constructor = eval(class_name)
                embeddings[key] = Constructor(key, **embed_args)

            model = cls.create(embeddings, **_state)
            model._state = _state
            model.load_weights(f"{basename}.wgt")

        return model

    @property
    def requires_state(self):
        return hasattr(self.impl, 'requires_state') and self.impl.requires_state


    def output(self, hidden, vsz, **kwargs):
        """Project to the output space

        :param hidden: (`tensor`) upstream layer
        :param vsz: (`int`) The vocab size
        :param kwargs: See below

        :Keyword Arguments:
        * *tie_weights* -- If weight tying is on, we are saying we want to use the (transposed) weights
                     declared for tgt embeddings.  Note that this nomenclature mostly makes sense if we are using
                     the tgt embeddings also as source embeddings.
                     If we are not, then having a `self.embeddings[self.tgt_key]` doesnt even make sense

        :return: Output
        """
        # Do weight sharing if we can
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        vocab_b = tf.get_variable("vocab_b", [vsz],  initializer=tf.zeros_initializer(), dtype=tf.float32)
        if do_weight_tying and self.hsz == self.embeddings[self.tgt_key].get_dsz():
            with tf.variable_scope(self.embeddings[self.tgt_key].scope, reuse=True):
                W = tf.get_variable("W")
            return tf.matmul(hidden, W, transpose_b=True, name="logits") + vocab_b
        else:
            vocab_w = tf.get_variable(
                "vocab_w", [self.hsz, vsz], dtype=tf.float32)
            return tf.nn.xw_plus_b(hidden, vocab_w, vocab_b, name="logits")


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):
    """RNN-based Language Model built on base class
    """
    def __init__(self):
        """Construct an RNNLM
        """
        super().__init__()
        self.rnntype = 'lstm'
        self.initial_state = None

    def decode(self, inputs, batchsz=1, rnntype='lstm', layers=1, **kwargs):
        """LSTM-based method for decoding

        :param inputs: The outputs of the embeddings
        :param batchsz: (`int`) The batch size
        :param rnntype: (`str`) What type of RNN (defaults to `lstm`)
        :param layers: (`int`) Defaults to 1
        :param variational: (`bool`) Using variational dropout?
        :param kwargs: See above

        :return: The layer
        """
        lstm_encoder_layer = LSTMEncoderWithState(None, self.hsz, layers, self.pdrop_value, **kwargs)
        self.initial_state = lstm_encoder_layer.zero_state(self.batchsz)
        inputs["h"] = self.initial_state
        return lstm_encoder_layer


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(LanguageModelBase):
    """Transformer-based Language Model built on base class
    """
    def __init__(self):
        """Construct an TLM
        """
        super().__init__()

    def decode(self, inputs, batchsz=1, num_heads=8, layers=1, **kwargs):
        """LSTM-based method for decoding

        :param inputs: The outputs of the embeddings
        :param batchsz: (`int`) The batch size
        :param rnntype: (`str`) What type of RNN (defaults to `lstm`)
        :param layers: (`int`) Defaults to 1
        :param variational: (`bool`) Using variational dropout?
        :param kwargs: See above

        :return: The layer
        """

        encoder_layer = TransformerEncoderStack(num_heads, self.hsz, self.pdrop_value, layers=layers, **kwargs)
        return encoder_layer


