import logging
from itertools import chain
from baseline.tf.seq2seq.encoders import *
from baseline.tf.seq2seq.decoders import *
from google.protobuf import text_format
from baseline.tf.tfy import *
from baseline.model import EncoderDecoderModel, register_model, create_seq2seq_decoder, create_seq2seq_encoder, create_seq2seq_arc_policy
from baseline.utils import ls_props, read_json, MAGIC_VARS
from baseline.tf.embeddings import *
from baseline.version import __version__
import copy

logger = logging.getLogger('baseline')


def _temporal_cross_entropy_loss(logits, labels, label_lengths, mx_seq_length):
    """Do cross-entropy loss accounting for sequence lengths

    :param logits: a `Tensor` with shape `[timesteps, batch, timesteps, vocab]`
    :param labels: an integer `Tensor` with shape `[batch, timesteps]`
    :param label_lengths: The actual length of the target text.  Assume right-padded
    :param mx_seq_length: The maximum length of the sequence
    :return:
    """

    # The labels actual length is 100, and starts with <GO>
    labels = tf.transpose(labels, perm=[1, 0])
    # TxB loss mask
    labels = labels[0:mx_seq_length, :]
    logit_length = tf.to_int32(tf.shape(logits)[0])
    timesteps = tf.to_int32(tf.shape(labels)[0])
    # The labels no longer include <GO> so go is not useful.  This means that if the length was 100 before, the length
    # of labels is now 99 (and that is the max allowed)
    pad_size = timesteps - logit_length
    logits = tf.pad(logits, [[0, pad_size], [0, 0], [0, 0]])
    #logits = logits[0:mx_seq_length, :, :]
    with tf.name_scope("Loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        # BxT loss mask
        loss_mask = tf.to_float(tf.sequence_mask(tf.to_int32(label_lengths), timesteps))
        # TxB losses * TxB loss_mask
        losses = losses * tf.transpose(loss_mask, [1, 0])

        losses = tf.reduce_sum(losses)
        losses /= tf.cast(tf.reduce_sum(label_lengths), tf.float32)
        return losses


class DataParallelEncoderDecoderModel(EncoderDecoderModel):

    def __init__(self, create_fn, src_embeddings, tgt_embedding, **kwargs):
        """Create N replica graphs for GPU + 1 for inference on CPU

        The basic idea of the constructor is to create several replicas for training by creating a `tf.split` operation
        on the input tensor and feeding the splits to each of the underlying replicas.  The way we do this is to take in
        the creation function for a single graph and call it N times while passing in the splits as kwargs.

        Because our `create_fn` (typically `cls.create()` where `cls` is a sub-class of EncoderDecoderModelBase) allows
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
        super(DataParallelEncoderDecoderModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus', -1)
        # If the gpu ID is set to -1, use CUDA_VISIBLE_DEVICES to figure it out
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        logger.info('Num GPUs %s', gpus)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()
        for key in src_embeddings.keys():
            EmbeddingType = src_embeddings[key].__class__
            self.parallel_params[key] = kwargs.get(key, EmbeddingType.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus, name='{}_split'.format(key))

        EmbeddingType = tgt_embedding.__class__
        self.parallel_params['tgt'] = kwargs.get('tgt', EmbeddingType.create_placeholder('tgt_parallel'))
        split_operations['tgt'] = tf.split(self.parallel_params['tgt'], gpus, name='tgt_split')

        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len_parallel"))
        src_len_splits = tf.split(self.src_len, gpus, name='src_len_split')
        split_operations['src_len'] = src_len_splits

        self.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len_parallel"))
        tgt_len_splits = tf.split(self.tgt_len, gpus, name='tgt_len_split')
        split_operations['tgt_len'] = tgt_len_splits

        self.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))

        losses = []
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

        with tf.device(tf.DeviceSpec(device_type="CPU")):
            # This change is a bit cleaner since we pop some fields in the sub model
            kwargs_infer = copy.deepcopy(kwargs)
            # This change is required since we attach our .x onto the object in v1
            # For the inference model, load it up on the CPU
            # This shares a sub-graph with its replicas after the inputs
            # It doesnt share the inputs, as these are placeholders, and the replicas are `tf.split` ops
            self.inference = create_fn(src_embeddings, tgt_embedding, sess=sess, **kwargs_infer)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                # Copy the input keyword args and update them
                kwargs_single = copy.deepcopy(kwargs)
                # For each split operator, there are N parts, take the part at index `i` and inject it for its key
                # this prevents the `create_fn` from making a placeholder for this operation
                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                # Create the replica
                replica = create_fn({k: v.detached_ref() for k, v in src_embeddings.items()},
                                    tgt_embedding.detached_ref(),
                                    sess=sess,
                                    mx_tgt_len=self.mx_tgt_len,
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
        feed_dict[self.tgt_len] = batch_dict['tgt_lengths']
        feed_dict[self.mx_tgt_len] = np.max(batch_dict['tgt_lengths'])

        for key in self.parallel_params.keys():
            feed_dict['{}_parallel:0'.format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict[self.src_len] = batch_dict[self.src_lengths_key]
        return feed_dict

    def load(self, basename, **kwargs):
        self.inference.load(basename, **kwargs)


class EncoderDecoderModelBase(EncoderDecoderModel):

    def create_loss(self):
        with tf.variable_scope('loss{}'.format(self.id)):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def create_test_loss(self):
        with tf.variable_scope('test_loss'):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def __init__(self):
        super(EncoderDecoderModelBase, self).__init__()
        self.saver = None
        self._unserializable = ['tgt']

    @classmethod
    @tf_device_wrapper
    def load(cls, basename, **kwargs):
        _state = read_json('{}.state'.format(basename))
        if __version__ != _state['version']:
            logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
        if 'predict' in kwargs:
            _state['predict'] = kwargs['predict']
        if 'beam' in kwargs:
            _state['beam'] = kwargs['beam']
        _state['sess'] = kwargs.get('sess', create_session())

        with _state['sess'].graph.as_default():

            src_embeddings_info = _state.pop('src_embeddings')
            src_embeddings = reload_embeddings(src_embeddings_info, basename)
            for k in src_embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            tgt_embedding_info = _state.pop('tgt_embedding')
            tgt_embedding = reload_embeddings(tgt_embedding_info, basename)['tgt']

            model = cls.create(src_embeddings, tgt_embedding, **_state)
            model._state = _state
            if kwargs.get('init', True):
                model.sess.run(tf.global_variables_initializer())
            model.saver = tf.train.Saver()
            model.saver.restore(model.sess, basename)
            return model

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_src = []
        for k, embedding in self.src_embeddings.items():
            x = kwargs.get(k, None)
            embeddings_out = embedding.encode(x)
            all_embeddings_src.append(embeddings_out)
        word_embeddings = tf.concat(values=all_embeddings_src, axis=-1)
        return word_embeddings

    def save_md(self, basename):
        write_json(self._state, '{}.state'.format(basename))
        for key, embedding in self.src_embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))
        self.tgt_embedding.save_md('{}-{}-md.json'.format(basename, 'tgt'))

    def _record_state(self, **kwargs):
        src_embeddings_info = {}
        for k, v in self.src_embeddings.items():
            src_embeddings_info[k] = v.__class__.__name__

        blacklist = set(chain(self._unserializable, MAGIC_VARS, self.src_embeddings.keys()))
        self._state = {k: v for k, v in kwargs.items() if k not in blacklist}
        self._state.update({
            'version': __version__,
            'module': self.__class__.__module__,
            'class': self.__class__.__name__,
            'src_embeddings': src_embeddings_info,
            'tgt_embedding': {'tgt': self.tgt_embedding.__class__.__name__}
        })

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        gpus = kwargs.get('gpus', 1)
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
            kwargs['gpus'] = gpus
        if gpus > 1:
            return DataParallelEncoderDecoderModel(cls.create, src_embeddings, tgt_embedding, **kwargs)
        model = cls()
        model.src_embeddings = src_embeddings
        model.tgt_embedding = tgt_embedding
        model._record_state(**kwargs)
        model.src_len = kwargs.pop('src_len', tf.placeholder(tf.int32, [None], name="src_len"))
        model.tgt_len = kwargs.pop('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len"))
        model.mx_tgt_len = kwargs.pop('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))
        model.src_lengths_key = kwargs.get('src_lengths_key')
        model.id = kwargs.get('id', 0)
        model.sess = kwargs.get('sess', create_session())
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.dropin_value = kwargs.get('dropin', {})
        model.layers = kwargs.get('layers', 1)
        model.hsz = kwargs['hsz']

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            embed_in = model.embed(**kwargs)
            encoder_output = model.encode(embed_in, **kwargs)
            model.decode(encoder_output, **kwargs)
            # writer = tf.summary.FileWriter('blah', model.sess.graph)
            return model

    def set_saver(self, saver):
        self.saver = saver

    @property
    def src_lengths_key(self):
        return self._src_lengths_key

    @src_lengths_key.setter
    def src_lengths_key(self, value):
        self._src_lengths_key = value

    def create_encoder(self, **kwargs):
        return create_seq2seq_encoder(**kwargs)

    def create_decoder(self, **kwargs):
        return create_seq2seq_decoder(self.tgt_embedding, **kwargs)

    def decode(self, encoder_output, **kwargs):
        self.decoder = self.create_decoder(**kwargs)
        predict = kwargs.get('predict', False)
        if predict:
            self.decoder.predict(encoder_output, self.src_len, self.pdrop_value, **kwargs)
        else:
            self.decoder.decode(encoder_output, self.src_len, self.tgt_len, self.pdrop_value, **kwargs)

    def encode(self, embed_in, **kwargs):
        with tf.variable_scope('encode'):
            self.encoder = self.create_encoder(**kwargs)
            return self.encoder.encode(embed_in, self.src_len, self.pdrop_value, **kwargs)

    @staticmethod
    def _write_props_to_state(obj, state):
        for prop in ls_props(obj):
            state[prop] = getattr(obj, prop)


    def save(self, model_base):
        self.save_md(model_base)
        self.saver.save(self.sess, model_base)

    def predict(self, batch_dict, **kwargs):
        feed_dict = self.make_input(batch_dict)
        vec = self.sess.run(self.decoder.best, feed_dict=feed_dict)
        # Vec is either [T, B] or [T, B, K]
        if len(vec.shape) == 2:
            # Add a fake K
            vec = np.expand_dims(vec, axis=2)
        # convert to (B x K x T)
        return vec.transpose(1, 2, 0)

    def step(self, batch_dict):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = self.make_input(batch_dict)
        x = self.sess.run(self.decoder.probs, feed_dict=feed_dict)
        return x


    @property
    def dropin_value(self):
        return self._dropin_value

    @dropin_value.setter
    def dropin_value(self, dict_value):
        self._dropin_value = dict_value

    def drop_inputs(self, key, x, do_dropout):
        v = self.dropin_value.get(key, 0)
        if do_dropout and v > 0.0:

            #do_drop = (np.random.random() < v)
            #if do_drop:
            #    drop_indices = np.where(x != Offsets.PAD)
            #    x[drop_indices[0], drop_indices[1]] = Offsets.PAD
            drop_indices = np.where((np.random.random(x.shape) < v) & (x != Offsets.PAD))
            x[drop_indices[0], drop_indices[1]] = Offsets.UNK
        return x

    def make_input(self, batch_dict, train=False):

        feed_dict = new_placeholder_dict(train)

        for key in self.src_embeddings.keys():
            feed_dict["{}:0".format(key)] = self.drop_inputs(key, batch_dict[key], train)

        if self.src_lengths_key is not None:
            feed_dict[self.src_len] = batch_dict[self.src_lengths_key]

        tgt = batch_dict.get('tgt')
        if tgt is not None:
            feed_dict["tgt:0"] = batch_dict['tgt']
            feed_dict[self.tgt_len] = batch_dict['tgt_lengths']
            feed_dict[self.mx_tgt_len] = np.max(batch_dict['tgt_lengths'])

        return feed_dict


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2Seq(EncoderDecoderModelBase):

    def __init__(self):
        super(Seq2Seq, self).__init__()
        self._vdrop = False

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value
