import logging
from itertools import chain
from baseline.tf.seq2seq.encoders import *
from baseline.tf.seq2seq.decoders import *
from baseline.tf.tfy import *
from baseline.tf.layers import new_placeholder_dict
from baseline.model import EncoderDecoderModel, register_model, create_seq2seq_decoder, create_seq2seq_encoder, create_seq2seq_arc_policy
from baseline.utils import ls_props, read_json, MAGIC_VARS
from baseline.tf.embeddings import *
from baseline.version import __version__

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


class EncoderDecoderModelBase(EncoderDecoderModel):

    def create_loss(self):
        with tf.variable_scope('loss'):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :],
                                                self.tgt_embedding.x[:, 1:],
                                                self.tgt_len - 1,
                                                self.mx_tgt_len - 1)

    def create_test_loss(self):
        with tf.variable_scope('test_loss'):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :],
                                                self.tgt_embedding.x[:, 1:],
                                                self.tgt_len - 1,
                                                self.mx_tgt_len - 1)

    def __init__(self):
        super(EncoderDecoderModelBase, self).__init__()
        self.saver = None
        self._unserializable = ['src_len', 'tgt_len', 'mx_tgt_len']

    def _record_state(self, **kwargs):
        self._state = {k: v for k, v in kwargs.items() if k not in self._unserializable + ['sess', 'tgt'] +
                       list(self.src_embeddings.keys())}
        src_embeddings_state = {}
        for k, v in self.src_embeddings.items():
            src_embeddings_state[k] = v.__class__.__name__  ##v._state

        self._state.update({
            "version": __version__,
            "src_embeddings": src_embeddings_state,
            "tgt_embedding": self.tgt_embedding.__class__.__name__  #self.tgt_embedding._state
        })

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
        state = {k: v for k, v in self._state.items()}

        write_json(state, '{}.state'.format(basename))
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
        model = cls()
        model.src_embeddings = {}
        for k, src_embedding in src_embeddings.items():
            model.src_embeddings[k] = src_embedding.detached_ref()
        model.tgt_embedding = tgt_embedding.detached_ref()
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
