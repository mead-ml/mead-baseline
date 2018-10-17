import tensorflow as tf
import json
from google.protobuf import text_format
from baseline.tf.tfy import *
import tensorflow.contrib.seq2seq as tfcontrib_seq2seq
from baseline.model import EncoderDecoderModel, register_model
from baseline.utils import ls_props, read_json
from baseline.tf.embeddings import *
from baseline.version import __version__
import copy


def _temporal_cross_entropy_loss(logits, labels, label_lengths, mx_seq_length):
    """Do cross-entropy loss accounting for sequence lengths
    
    :param logits: a `Tensor` with shape `[timesteps, batch, vocab]`
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


class Seq2SeqParallelModel(EncoderDecoderModel):

    def __init__(self, create_fn, src_embeddings, tgt_embedding, **kwargs):
        super(Seq2SeqParallelModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus', -1)
        # If the gpu ID is set to -1, use CUDA_VISIBLE_DEVICES to figure it out
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        print('Num GPUs', gpus)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()
        for key in src_embeddings.keys():
            EmbeddingType = src_embeddings[key].__class__
            self.parallel_params[key] = kwargs.get(key, EmbeddingType.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus)

        EmbeddingType = tgt_embedding.__class__
        self.parallel_params['tgt'] = kwargs.get(key, EmbeddingType.create_placeholder('tgt_parallel'.format(key)))
        split_operations['tgt'] = tf.split(self.parallel_params[key], gpus)

        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len_parallel"))
        src_len_splits = tf.split(self.src_len, gpus)
        split_operations['src_len'] = src_len_splits

        self.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len_parallel"))
        tgt_len_splits = tf.split(self.tgt_len, gpus)
        split_operations['tgt_len'] = tgt_len_splits

        self.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))
        self.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, (), name="pkeep"))
        self.pdrop_value = kwargs.get('dropout', 0.5)

        losses = []
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            self.inference = create_fn(src_embeddings, tgt_embedding, sess=sess, mx_tgt_len=self.mx_tgt_len, pkeep=self.pkeep, id=1, **kwargs)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                kwargs_single = copy.deepcopy(kwargs)
                kwargs_single['sess'] = sess
                kwargs_single['pkeep'] = self.pkeep
                kwargs_single['id'] = i + 1
                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                replica = create_fn(src_embeddings, tgt_embedding, **kwargs_single)
                self.replicas.append(replica)
                loss_op = replica.create_loss()
                losses.append(loss_op)

        self.loss = tf.reduce_mean(tf.stack(losses))

        self.sess = sess
        self.best = self.inference.best

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

    def make_input(self, batch_dict, do_dropout=False):
        if do_dropout is False:
            return self.inference.make_input(batch_dict)

        tgt = batch_dict.get['tgt']
        tgt_len = batch_dict['tgt_len']
        mx_tgt_len = np.max(tgt_len)
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep, "tgt:0": tgt, self.tgt_len: tgt_len, self.mx_tgt_len: mx_tgt_len}

        for key in self.parallel_params.keys():
            feed_dict["{}_parallel:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict[self.src_len] = batch_dict[self.src_lengths_key]
        return feed_dict

    def load(self, basename, **kwargs):
        self.inference.load(basename, **kwargs)


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2SeqModel(EncoderDecoderModel):

    def create_loss(self):
        with tf.variable_scope('Loss{}'.format(self.id), reuse=False):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def create_test_loss(self):
        with tf.variable_scope('Loss', reuse=False):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.saver = None

    @staticmethod
    def load(basename, **kwargs):
        state = read_json(basename + '.state')
        if 'predict' in kwargs:
            state['predict'] = kwargs['predict']

        if 'sampling' in kwargs:
            state['sampling'] = kwargs['sampling']

        if 'sampling_temp' in kwargs:
            state['sampling_temp'] = kwargs['sampling_temp']

        if 'beam' in kwargs:
            state['beam'] = kwargs['beam']

        state['sess'] = kwargs.get('sess', tf.Session())

        if 'model_type' in kwargs:
            state['model_type'] = kwargs['model_type']
        elif state['attn']:
            print('setting to attn')
            state['model_type'] = 'attn' if state['attn'] is True else 'default'

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        src_embeddings = dict()
        src_embeddings_dict = state.pop('src_embeddings')
        for key, class_name in src_embeddings_dict.items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            src_embeddings[key] = Constructor(key, **embed_args)

        tgt_class_name = state.pop('tgt_embedding')
        md = read_json('{}-tgt-md.json'.format(basename))
        embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
        Constructor = eval(tgt_class_name)
        tgt_embedding = Constructor('tgt', **embed_args)
        model = Seq2SeqModel.create(src_embeddings, tgt_embedding, **state)
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])
        do_init = kwargs.get('init', True)
        if do_init:
            init = tf.global_variables_initializer()
            model.sess.run(init)

        model.saver = tf.train.Saver()
        model.saver.restore(model.sess, basename)
        return model

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):

        gpus = kwargs.get('gpus')
        if gpus is not None:
            return Seq2SeqParallelModel(Seq2SeqModel.create, src_embeddings, tgt_embedding, **kwargs)
        model = cls()
        model.src_embeddings = src_embeddings
        model.tgt_embedding = tgt_embedding
        model.tgt_embedding.x = tgt_embedding.create_placeholder(tgt_embedding.name)
        GO = kwargs.get('GO')
        EOS = kwargs.get('EOS')
        model.GO = GO
        model.EOS = EOS
        model.id = kwargs.get('id', 0)
        model.src_lengths_key = kwargs.get('src_lengths_key')
        model.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len"))
        model.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len"))
        #model.tgt = kwargs.get('tgt', tf.placeholder(tf.int32, [None, None], name="tgt"))
        model.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))

        hsz = int(kwargs['hsz'])
        attn = kwargs.get('model_type') == 'attn'
        layers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'lstm')
        predict = kwargs.get('predict', False)
        beam_width = kwargs.get('beam', 1) if predict is True else 1
        sampling = kwargs.get('sampling', False)
        sampling_temp = kwargs.get('sampling_temp', 1.0)
        model.sess = kwargs.get('sess', tf.Session())
        model.vdrop = bool(kwargs.get('variational_dropout', False))
        unif = kwargs.get('unif', 0.25)

        model.pdrop_value = kwargs.get('dropout', 0.5)

        model.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))
        attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        model.arc_state = kwargs.get('arc_state', False)
        model.hsz = hsz
        model.layers = layers
        model.rnntype = rnntype
        model.attn = attn

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

            all_embeddings_src = []
            for embedding in model.src_embeddings.values():
                embeddings_out = embedding.encode()
                all_embeddings_src.append(embeddings_out)

            embed_in = tf.concat(values=all_embeddings_src, axis=-1)

            # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
            # which makes it really hard to get the same objects between train and test
            # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
            # This works fine for training, but then at decode time its not quite in the right place scope-wise
            # So instead, for now, we never call .encode() and instead we create our own operator
            Wo = tf.get_variable("Wo", initializer=tf.constant_initializer(tgt_embedding.weights,
                                                                           dtype=tf.float32,
                                                                           verify_shape=True),
                                 shape=[tgt_embedding.vsz, tgt_embedding.dsz])

            with tf.variable_scope("Recurrence"):
                rnn_enc_tensor, final_encoder_state = model.encode(embed_in)
                batch_sz = tf.shape(rnn_enc_tensor)[0]
                with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
                    proj = dense_layer(tgt_embedding.vsz)
                    rnn_dec_cell = model._attn_cell_w_dropout(rnn_enc_tensor, beam_width, attn_type)

                    if beam_width > 1:
                        final_encoder_state = tf.contrib.seq2seq.tile_batch(final_encoder_state, multiplier=beam_width)

                    if model.attn is True:
                        initial_state = rnn_dec_cell.zero_state(batch_sz*beam_width, tf.float32)
                        if model.arc_state is True:
                            initial_state = initial_state.clone(cell_state=final_encoder_state)
                    else:
                        initial_state = final_encoder_state

                    if predict is True:
                        if beam_width == 1:
                            if sampling:
                                helper = tf.contrib.seq2seq.SamplingEmbeddingHelper(Wo, tf.fill([batch_sz], GO), EOS,
                                                                                    softmax_temperature=sampling_temp)
                            else:
                                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(Wo, tf.fill([batch_sz], GO), EOS)
                            decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper,
                                                                      initial_state=initial_state, output_layer=proj)
                        else:

                            # Define a beam-search decoder
                            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                cell=rnn_dec_cell,
                                embedding=Wo,
                                start_tokens=tf.fill([batch_sz], GO),
                                end_token=EOS,
                                initial_state=initial_state,
                                beam_width=beam_width,
                                output_layer=proj,
                                length_penalty_weight=0.0)
                    else:
                        helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, tgt_embedding.x), sequence_length=model.tgt_len)
                        decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper, initial_state=initial_state, output_layer=proj)

                    # This creates a "decoder" scope
                    final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                              impute_finished=predict is False or beam_width == 1,
                                                                                              swap_memory=True,
                                                                                              output_time_major=True)
                    if predict is True and beam_width > 1:
                        model.preds = tf.no_op()
                        best = final_outputs.predicted_ids
                    else:
                        model.preds = final_outputs.rnn_output
                        best = final_outputs.sample_id
            with tf.variable_scope("Output"):
                model.best = tf.identity(best, name='best')
                if beam_width > 1:
                    model.probs = tf.no_op(name='probs')
                else:
                    model.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), model.preds)

            # writer = tf.summary.FileWriter('blah', model.sess.graph)
            ##model.GO = GO
            ##model.EOS = EOS
            return model

    def set_saver(self, saver):
        self.saver = saver

    @property
    def src_lengths_key(self):
        return self._src_lengths_key

    @src_lengths_key.setter
    def src_lengths_key(self, value):
        self._src_lengths_key = value

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def _attn_cell_w_dropout(self, rnn_enc_tensor, beam, attn_type):
        cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, self.layers, variational=self.vdrop)
        if self.attn:
            src_len = self.src_len
            if beam > 1:
                # Expand the encoded tensor for all beam entries
                rnn_enc_tensor = tf.contrib.seq2seq.tile_batch(rnn_enc_tensor, multiplier=beam)
                src_len = tf.contrib.seq2seq.tile_batch(src_len, multiplier=beam)
            GlobalAttention = tfcontrib_seq2seq.LuongAttention if attn_type == 'luong' else tfcontrib_seq2seq.BahdanauAttention
            attn_mech = GlobalAttention(self.hsz, rnn_enc_tensor, src_len)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
        return cell

    def encode(self, embed_in):
        with tf.variable_scope('encode'):
            # List to tensor, reform as (T, B, W)
            if self.rnntype == 'blstm':

                nlayers_bi = int(self.layers / 2)
                rnn_fwd_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, nlayers_bi, variational=self.vdrop)
                rnn_bwd_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, nlayers_bi, variational=self.vdrop)
                rnn_enc_tensor, final_encoder_state = tf.nn.bidirectional_dynamic_rnn(rnn_fwd_cell, rnn_bwd_cell,
                                                                                      embed_in,
                                                                                      scope='brnn_enc',
                                                                                      sequence_length=self.src_len,
                                                                                      dtype=tf.float32)
                rnn_enc_tensor = tf.concat(rnn_enc_tensor, -1)
                encoder_state = []
                for i in range(nlayers_bi):
                    encoder_state.append(final_encoder_state[0][i])  # forward
                    encoder_state.append(final_encoder_state[1][i])  # backward
                encoder_state = tuple(encoder_state)
            else:

                rnn_enc_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, self.layers, variational=self.vdrop)
                rnn_enc_tensor, encoder_state = tf.nn.dynamic_rnn(rnn_enc_cell, embed_in,
                                                                  scope='rnn_enc',
                                                                  sequence_length=self.src_len,
                                                                  dtype=tf.float32)

            # This comes out as a sequence T of (B, D)
            return rnn_enc_tensor, encoder_state

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])
        src_embeddings_info = {}
        for k, v in self.src_embeddings.items():
            src_embeddings_info[k] = v.__class__.__name__
        state = {
            "version": __version__,
            "src_embeddings": src_embeddings_info,
            "tgt_embedding": self.tgt_embedding.__class__.__name__,
            "attn": self.attn,
            "hsz": self.hsz,
            "rnntype": self.rnntype,
            "layers": self.layers,
            "arc_state": self.arc_state,
            "EOS": self.EOS,
            "GO": self.GO
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        for key, embedding in self.src_embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

        self.tgt_embedding.save_md('{}-tgt-md.json'.format(basename))

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

    def save(self, model_base):
        self.save_md(model_base)
        self.saver.save(self.sess, model_base)

    def restore_graph(self, base):
        with open(base + '.graph', 'rb') as gf:
            gd = tf.GraphDef()
            gd.ParseFromString(gf.read())
            self.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

    def run(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        vec = self.sess.run(self.best, feed_dict=feed_dict)
        # (B x K x T)
        if len(vec.shape) == 3:
            return vec.transpose(1, 2, 0)
        else:
            return vec.transpose(1, 0)

    def step(self, batch_dict):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = self.make_input(batch_dict)
        return self.sess.run(self.probs, feed_dict=feed_dict)

    def make_input(self, batch_dict, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.src_embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        if self.src_lengths_key is not None:
            feed_dict[self.src_len] = batch_dict[self.src_lengths_key]

        tgt = batch_dict.get('tgt')
        if tgt is not None:
            feed_dict["tgt:0"] = batch_dict['tgt']
            feed_dict[self.tgt_len] = batch_dict['tgt_lengths']
            feed_dict[self.mx_tgt_len] = np.max(batch_dict['tgt_lengths'])

        return feed_dict
