import logging
from itertools import chain
from baseline.tf.tfy import *
from baseline.version import __version__
from baseline.model import LanguageModel, register_model
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_encoder_stack, subsequent_mask
from baseline.utils import read_json, write_json, ls_props, MAGIC_VARS
from google.protobuf import text_format
import copy


logger = logging.getLogger('baseline')


class DataParallelLanguageModel(LanguageModel):

    def __init__(self, create_fn, embeddings, **kwargs):
        """Create N replica graphs for GPU + 1 for inference on CPU

        The basic idea of the constructor is to create several replicas for training by creating a `tf.split` operation
        on the input tensor and feeding the splits to each of the underlying replicas.  The way we do this is to take in
        the creation function for a single graph and call it N times while passing in the splits as kwargs.

        Because our `create_fn` (typically `cls.create()` where `cls` is a sub-class of LanguageModelBase) allows
        us to inject its inputs through keyword arguments instead of creating its own placeholders, we can inject each
        split into the inputs which causes each replica to be a sub-graph of this parent graph.  For this to work,
        this class also has to have its own placeholders, which it uses as inputs.

        Any time we are doing validation during the training process, we delegate the request to the underlying member
        variable `.inference` (which is sharing its weights with the other replicas).  This also happens on `save()`,
        allowing us to create a perfectly normal single sub-graph checkpoint for later inference.

        The actual way that we accomplish the replica creation is by copying the input keyword arguments and injecting
        any parallel operations (splits) by deep-copy and update to the dictionary.

        :param create_fn: This function is actually our caller, who creates the graph (if no `gpus` arg)
        :param src_embeddings: This is the set of src embeddings sub-graphs
        :param tgt_embedding: This is the tgt embeddings sub-graph
        :param kwargs: See below, also see ``EncoderDecoderModelBase.create`` for kwargs that are not specific to multi-GPU

        :Keyword Arguments:
        * *gpus* -- (``int``) - The number of GPUs to create replicas on
        * *src_lengths_key* -- (``str``) - A string representing the key for the src tensor whose lengths should be used
        * *mx_tgt_len* -- (``int``) - An optional max length (or we will use the max length of the batch using a placeholder)

        """
        super(DataParallelLanguageModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus')
        logger.info('Num GPUs %s', gpus)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()

        self.tgt_key = kwargs.get('tgt_key')
        if self.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')

        for key in embeddings.keys():
            EmbeddingType = embeddings[key].__class__
            self.parallel_params[key] = kwargs.get(key, EmbeddingType.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus, name='{}_split'.format(key))

            if self.tgt_key == key:
                self.parallel_params['y'] = kwargs.get('y', EmbeddingType.create_placeholder('y_parallel'))
                y_splits = tf.split(self.parallel_params['y'], gpus)
                split_operations['y'] = y_splits

        losses = []
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        with tf.device(tf.DeviceSpec(device_type="CPU")):
            # This change is a bit cleaner since we pop some fields in the sub model
            kwargs_infer = copy.deepcopy(kwargs)
            # This change is required since we attach our .x onto the object in v1
            # For the inference model, load it up on the CPU
            # This shares a sub-graph with its replicas after the inputs
            # It doesnt share the inputs, as these are placeholders, and the replicas are `tf.split` ops
            self.inference = create_fn(embeddings, sess=sess, **kwargs_infer)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                # Copy the input keyword args and update them
                kwargs_single = copy.deepcopy(kwargs)
                # For each split operator, there are N parts, take the part at index `i` and inject it for its key
                # this prevents the `create_fn` from making a placeholder for this operation
                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                # Create the replica
                replica = create_fn({k: v.detached_ref() for k, v in embeddings.items()},
                                    sess=sess,
                                    id=i+1,
                                    **kwargs_single)
                # Add the replica to the set
                self.replicas.append(replica)
                # Make a replica specific loss
                loss_op = replica.create_loss()
                # Append to losses
                losses.append(loss_op)

        # The training loss is the mean of all replica losses
        self.loss = tf.reduce_mean(tf.stack(losses))

        self.sess = sess
        ##self.best = self.inference.best

    def create_loss(self):
        return self.loss

    def create_test_loss(self):
        return self.inference.create_test_loss()

    def save(self, model_base):
        return self.inference.save(model_base)

    def set_saver(self, saver):
        self.inference.saver = saver
        self.saver = saver

    def step(self, batch_dict):
        """
        Generate probability distribution over output V for next token
        """
        return self.inference.step(batch_dict)

    def make_input(self, batch_dict, train=False):
        if train is False:
            return self.inference.make_input(batch_dict)

        feed_dict = new_placeholder_dict(train)
        for key in self.parallel_params.keys():
            feed_dict['{}_parallel:0'.format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict['y_parallel:0'] = batch_dict['y']
        return feed_dict

    def load(self, basename, **kwargs):
        self.inference.load(basename, **kwargs)


class LanguageModelBase(LanguageModel):

    def __init__(self):
        self.saver = None
        self.layers = None
        self.hsz = None
        self.probs = None
        self._unserializable = []

    def set_saver(self, saver):
        self.saver = saver

    def decode(self, inputs):
        pass

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def save_md(self, basename):
        write_json(self._state, '{}.state'.format(basename))
        for key, embedding in self.embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

    def _record_state(self, **kwargs):
        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(self._unserializable, MAGIC_VARS, self.embeddings.keys()))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'embeddings': embeddings_info,
        })

    def _create_loss(self, scope):

        with tf.variable_scope(scope):
            vsz = self.embeddings[self.tgt_key].vsz
            targets = tf.reshape(self.y, [-1])
            outputs = tf.reshape(self.logits, [-1, vsz])
            bt_x_v = tf.nn.log_softmax(outputs, axis=-1)
            one_hots = tf.one_hot(targets, vsz)
            example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
            #example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #    logits=outputs,
            #    labels=targets
            #)
            loss = tf.reduce_mean(example_loss)
            return loss

    def create_loss(self):
        return self._create_loss(scope='loss{}'.format(self.id))

    def create_test_loss(self):
        return self._create_loss(scope='test_loss')

    def make_input(self, batch_dict, train=False):

        feed_dict = new_placeholder_dict(train)

        for key in self.src_keys:
            feed_dict["{}:0".format(key)] = batch_dict[key]

        y = batch_dict.get('y')
        if y is not None:
            feed_dict[self.y] = batch_dict['y']

        return feed_dict

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        step_softmax = self.sess.run(self.probs, feed_dict)
        return step_softmax

    @classmethod
    def create(cls, embeddings, **kwargs):
        gpus = kwargs.get('gpus', 1)
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
            kwargs['gpus'] = gpus
        if gpus > 1:
            return DataParallelLanguageModel(cls.create, embeddings, **kwargs)
        lm = cls()
        lm.id = kwargs.get('id', 0)
        lm.embeddings = {k: embedding.detached_ref() for k, embedding in embeddings.items()}
        lm._record_state(**kwargs)
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        lm.sess = kwargs.get('sess', create_session())
        lm.pdrop_value = kwargs.get('pdrop', 0.5)
        lm.hsz = kwargs['hsz']
        lm.src_keys = kwargs.get('src_keys', lm.embeddings.keys())
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')
        unif = kwargs.get('unif', 0.05)
        weight_initializer = tf.random_uniform_initializer(-unif, unif)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE, initializer=weight_initializer):

            inputs = lm.embed(**kwargs)
            lm.layers = kwargs.get('layers', 1)
            h = lm.decode(inputs, **kwargs)
            lm.logits = lm.output(h, lm.embeddings[lm.tgt_key].vsz, **kwargs)
            lm.probs = tf.nn.softmax(lm.logits, name="softmax")

            return lm

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_src = []
        for k in self.src_keys:
            embedding = self.embeddings[k]
            x = kwargs.get(k, None)
            embeddings_out = embedding.encode(x)
            all_embeddings_src.append(embeddings_out)
        word_embeddings = tf.concat(values=all_embeddings_src, axis=-1)
        embed_output = tf.layers.dropout(word_embeddings, rate=self.pdrop_value, training=TRAIN_FLAG())
        projsz = kwargs.get('projsz')
        if projsz:
            embed_output = tf.layers.dense(embed_output, projsz)
        return embed_output

    @classmethod
    @tf_device_wrapper
    def load(cls, basename, **kwargs):
        _state = read_json('{}.state'.format(basename))
        if __version__ != _state['version']:
            logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        if 'predict' in kwargs:
            _state['predict'] = kwargs['predict']
        if 'sampling' in kwargs:
            _state['sampling'] = kwargs['sampling']
        if 'sampling_temp' in kwargs:
            _state['sampling_temp'] = kwargs['sampling_temp']
        if 'beam' in kwargs:
            _state['beam'] = kwargs['beam']
        _state['sess'] = kwargs.get('sess', create_session())

        with _state['sess'].graph.as_default():

            embeddings_info = _state.pop('embeddings')
            embeddings = reload_embeddings(embeddings_info, basename)
            for k in embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            model = cls.create(embeddings, **_state)
            if kwargs.get('init', True):
                model.sess.run(tf.global_variables_initializer())
            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
            return model

    def output(self, h, vsz, **kwargs):
        # Do weight sharing if we can
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        vocab_b = tf.get_variable("vocab_b", [vsz],  initializer=tf.zeros_initializer(), dtype=tf.float32)
        if do_weight_tying and self.hsz == self.embeddings[self.tgt_key].get_dsz():
            with tf.variable_scope(self.embeddings[self.tgt_key].scope, reuse=True):
                W = tf.get_variable("W")
            return tf.matmul(h, W, transpose_b=True, name="logits") + vocab_b
        else:
            vocab_w = tf.get_variable(
                "vocab_w", [self.hsz, vsz], dtype=tf.float32)
            return tf.nn.xw_plus_b(h, vocab_w, vocab_b, name="logits")


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):
    def __init__(self):
        super(RNNLanguageModel, self).__init__()
        self.rnntype = 'lstm'

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def decode(self, inputs, rnntype='lstm', variational_dropout=False, **kwargs):

        def cell():
            skip_conn = bool(kwargs.get('skip_conn', False))
            projsz = kwargs.get('projsz')
            return lstm_cell_w_dropout(self.hsz, self.pdrop_value, variational=self.vdrop, training=TRAIN_FLAG(),
                                       skip_conn=skip_conn, projsz=projsz)

        self.rnntype = rnntype
        self.vdrop = variational_dropout

        rnnfwd = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.layers)], state_is_tuple=True)
        self.initial_state = rnnfwd.zero_state(tf.shape(inputs)[0], tf.float32)
        rnnout, state = tf.nn.dynamic_rnn(rnnfwd, inputs, initial_state=self.initial_state, dtype=tf.float32)
        h = tf.reshape(tf.concat(rnnout, 1), [-1, self.hsz])
        self.final_state = state
        return h


@register_model(task='lm', name='transformer')
class TransformerLanguageModel(LanguageModelBase):
    def __init__(self):
        super(TransformerLanguageModel, self).__init__()

    def decode(self, x, num_heads=4, layers=1, scale=True, activation_type='relu', scope='TransformerEncoder', d_ff=None, **kwargs):
        T = get_shape_as_list(x)[1]
        dsz = get_shape_as_list(x)[-1]
        mask = subsequent_mask(T)
        if dsz != self.hsz:
            x = tf.layers.dense(x, self.hsz)
        x = transformer_encoder_stack(x, mask, num_heads, self.pdrop_value, scale, layers, activation_type, d_ff=d_ff)
        return tf.reshape(x, [-1, self.hsz])


@register_model(task='lm', name='transformer-mlm')
class TransformerMaskedLanguageModel(LanguageModelBase):
    def __init__(self):
        super(TransformerMaskedLanguageModel, self).__init__()

    def decode(self, x, num_heads=4, layers=1, scale=True, activation_type='relu', scope='TransformerEncoder', d_ff=None, **kwargs):
        T = get_shape_as_list(x)[1]
        dsz = get_shape_as_list(x)[-1]
        mask = tf.ones([1, 1, T, T])
        if dsz != self.hsz:
            x = tf.layers.dense(x, self.hsz)
        x = transformer_encoder_stack(x, mask, num_heads, self.pdrop_value, scale, layers, activation_type, d_ff=d_ff)
        return tf.reshape(x, [-1, self.hsz])
