import logging
from itertools import chain
from baseline.tf.seq2seq.encoders import *
from baseline.tf.seq2seq.decoders import *
from baseline.tf.tfy import *
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
    # labels = tf.Print(labels, [tf.shape(labels)], message="Label Shape: ")
    # logits = tf.Print(logits, [tf.shape(logits)], message="Logits Shape: ")
    labels = tf.transpose(labels, perm=[1, 0])
    # labels = tf.Print(labels, [tf.shape(labels)], message="Label.T Shape: ")
    # TxB loss mask
    labels = labels[:mx_seq_length, :]
    # labels = tf.Print(labels, [tf.shape(labels)], message="Label cut Shape: ")
    logits = logits[:mx_seq_length]
    # logits = tf.Print(logits, [tf.shape(logits)], message="logits cut Shape: ")
    logit_length = tf.to_int32(tf.shape(logits)[0])
    # logit_length = tf.Print(logit_length, [logit_length], message='Length of logits')
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
        loss_mask = tf.cast(tf.sequence_mask(tf.cast(label_lengths, tf.int32), timesteps), tf.float32)
        # TxB losses * TxB loss_mask
        losses = losses * tf.transpose(loss_mask, [1, 0])

        losses = tf.reduce_sum(losses)
        losses /= tf.cast(tf.reduce_sum(label_lengths), tf.float32)
        return losses

if not tf.executing_eagerly():
    class EncoderDecoderModelBase(EncoderDecoderModel, tf.keras.Model):

        def __init__(self, name=None):
            super().__init__(name=name)

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
            super().__init__()
            self.saver = None
            self._unserializable = ['src_len', 'tgt_len', 'mx_tgt_len']

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

                model = cls.create(src_embeddings, tgt_embedding, reload=True, **_state)
                model._state = _state
                if kwargs.get('init', True):
                    model.sess.run(tf.compat.v1.global_variables_initializer())
                model.saver = tf.compat.v1.train.Saver()
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
                model.src_embeddings[k] = src_embedding  #.detached_ref()
            model.tgt_embedding = tgt_embedding  #.detached_ref()
            model._record_state(**kwargs)
            model.src_len = kwargs.pop('src_len', tf.compat.v1.placeholder(tf.int32, [None], name="src_len"))
            model.tgt_len = kwargs.pop('tgt_len', tf.compat.v1.placeholder(tf.int32, [None], name="tgt_len"))
            model.mx_tgt_len = kwargs.pop('mx_tgt_len', tf.compat.v1.placeholder(tf.int32, name="mx_tgt_len"))
            model.src_lengths_key = kwargs.get('src_lengths_key')
            model.id = kwargs.get('id', 0)
            model.sess = kwargs.get('sess', create_session())
            model.pdrop_value = kwargs.get('dropout', 0.5)
            model.dropin_value = kwargs.get('dropin', {})
            model.num_layers = kwargs.get('layers', 1)
            model.hsz = kwargs['hsz']

            embed_in = model.embed(**kwargs)
            encoder_output = model.encode(embed_in, **kwargs)
            model.decode(encoder_output, **kwargs)
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
            self.decoder = self.create_decoder(pdrop=self.pdrop_value, **kwargs)
            predict = kwargs.get('predict', False)
            if predict:
                self.decoder((encoder_output, self.src_len), **kwargs)
            else:
                tgt = kwargs.get('tgt')
                self.decoder((encoder_output, tgt, self.src_len, self.tgt_len), **kwargs)

        def encode(self, embed_in, **kwargs):
            self.encoder = self.create_encoder(pdrop=self.pdrop_value, **kwargs)
            return self.encoder((embed_in, self.src_len), **kwargs)

        def save(self, model_base):
            self.save_md(model_base)
            self.saver.save(self.sess, model_base, write_meta_graph=False)

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
                feed_dict[self.tgt_embedding.x] = batch_dict['tgt']
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


else:
    logger = logging.getLogger('baseline')


    class EncoderDecoderModelBase(tf.keras.Model, EncoderDecoderModel):

        def __init__(self, src_embeddings, tgt_embedding, **kwargs):
            super().__init__()
            self.beam_sz = kwargs.get('beam', 1)
            self._unserializable = []
            # These will get reinitialized as Layers
            # TODO: shouldnt need this if we record later
            self.src_embeddings = src_embeddings
            self.tgt_embedding = tgt_embedding

            self._record_state(**kwargs)
            src_dsz = self.init_embed(src_embeddings, tgt_embedding)
            self.dropin_values = kwargs.get('dropin', {})
            self.encoder = self.init_encoder(src_dsz, **kwargs)
            self.decoder = self.init_decoder(tgt_embedding, **kwargs)

        def init_embed(self, src_embeddings, tgt_embedding, **kwargs):
            """This is the hook for providing embeddings.  It takes in a dictionary of `src_embeddings` and a single
            tgt_embedding` of type `PyTorchEmbedding`

            :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings, one per embedding
            :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
            :param kwargs:
            :return: Return the aggregate embedding input size
            """
            if isinstance(src_embeddings, dict):
                self.src_embeddings = EmbeddingsStack(src_embeddings)
            else:
                assert isinstance(src_embeddings, EmbeddingsStack)
                self.src_embeddings = src_embeddings
            self.tgt_embedding = tgt_embedding
            return self.src_embeddings

        def init_encoder(self, input_sz, **kwargs):
            # This is a hack since TF never needs this one, there is not a general constructor param, so shoehorn
            kwargs['dsz'] = input_sz
            return create_seq2seq_encoder(**kwargs)

        def init_decoder(self, tgt_embedding, **kwargs):
            return create_seq2seq_decoder(tgt_embedding, **kwargs)

        def encode(self, input, lengths):
            """

            :param input:
            :param lengths:
            :return:
            """
            embed_in_seq = self.embed(input)
            return self.encoder((embed_in_seq, lengths))

        def decode(self, encoder_outputs, dst):
            return self.decoder(encoder_outputs, dst)

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

        def save(self, model_base):
            self.save_md(model_base)
            self.save_values(model_base)

        def save_values(self, basename):
            """Save tensor files out

            :param basename: Base name of model
            :return:
            """
            self.save_weights(f"{basename}.wgt")

        @classmethod
        @tf_device_wrapper
        def load(cls, basename, **kwargs):
            import pdb; pdb.set_trace()
            _state = read_json('{}.state'.format(basename))
            if __version__ != _state['version']:
                logger.warning("Loaded model is from baseline version %s, running version is %s", _state['version'], __version__)
            if 'predict' in kwargs:
                _state['predict'] = kwargs['predict']
            if 'beam' in kwargs:
                _state['beam'] = kwargs['beam']


            src_embeddings_info = _state.pop('src_embeddings')
            src_embeddings = reload_embeddings(src_embeddings_info, basename)
            for k in src_embeddings_info:
                if k in kwargs:
                    _state[k] = kwargs[k]
            tgt_embedding_info = _state.pop('tgt_embedding')
            tgt_embedding = reload_embeddings(tgt_embedding_info, basename)['tgt']

            model = cls.create(src_embeddings, tgt_embedding, **_state)
            model._state = _state
            model.load_weights(f"{basename}.wgt")
            return model


        @classmethod
        def create(cls, src_embeddings, tgt_embedding, **kwargs):
            model = cls(src_embeddings, tgt_embedding, **kwargs)
            logger.info(model)
            return model

        #def drop_inputs(self, key, x):
        #    v = self.dropin_values.get(key, 0)
        #
        #    if not TRAIN_FLAG() or v == 0:
        #        return x
        #
        #    mask_pad = x != Offsets.PAD
        #    mask_drop = x.new(x.size(0), x.size(1)).bernoulli_(v).byte()
        #    x.masked_fill_(mask_pad & mask_drop, Offsets.UNK)
        #    return x

        def embed(self, input):
            return self.src_embeddings(input)

        def call(self, input):
            src_len = input['src_len']
            encoder_outputs = self.encode(input, src_len)
            output = self.decode(encoder_outputs, input['dst'])
            # Return as B x T x H
            return output

        def predict(self, inputs, **kwargs):
            """Predict based on the batch.

            If `make_input` is True then run make_input on the batch_dict.
            This is false for being used during dev eval where the inputs
            are already transformed.
            """
            SET_TRAIN_FLAG(False)
            encoder_outputs = self.encode(inputs, inputs['src_len'])
            #outs = self.greedy_decode(encoder_outputs, **kwargs)
            #return outs
            outs, lengths, scores = self.decoder.beam_search(encoder_outputs, **kwargs)
            return outs.numpy()

        def greedy_decode(self, encoder_outputs, **kwargs):
            mxlen = kwargs.get('mxlen', 100)
            B = get_shape_as_list(encoder_outputs.output)[0]
            #log_probs = np.zeros(B)
            paths = np.full((B, 1, 1), Offsets.GO)

            src_mask = encoder_outputs.src_mask
            h_i, dec_out, context = self.decoder.arc_policy((encoder_outputs, self.decoder.hsz, 1))

            for i in range(mxlen - 1):
                """Calculate the probs of the next output and update state."""
                # Our RNN decoder is now batch-first, so we need to expand the time dimension
                last = tf.reshape(paths[:, :, -1], (-1, 1))
                dec_out, h_i = self.decoder.decode_rnn(context, h_i, dec_out, last, src_mask)
                probs = self.decoder.output(dec_out)
                # Collapse over time
                dec_out = tf.squeeze(dec_out, 1)
                best = tf.argmax(probs, -1)
                paths = tf.concat([paths, tf.expand_dims(best, -1)], axis=2)

            return paths[:, :, 1:]

    @register_model(task='seq2seq', name=['default', 'attn'])
    class Seq2SeqModel(EncoderDecoderModelBase):

        def __init__(self, src_embeddings, tgt_embedding, **kwargs):
            """This base model is extensible for attention and other uses.  It declares minimal fields allowing the
            subclass to take over most of the duties for drastically different implementations

            :param src_embeddings: (``dict``) A dictionary of PyTorchEmbeddings
            :param tgt_embedding: (``PyTorchEmbeddings``) A single PyTorchEmbeddings object
            :param kwargs:
            """
            super().__init__(src_embeddings, tgt_embedding, **kwargs)
