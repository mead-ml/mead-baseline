from baseline.tf.tfy import *
import tensorflow.contrib.seq2seq as tfcontrib_seq2seq
from baseline.utils import ls_props, read_json, Offsets, export
from baseline.model import register_decoder, register_arc_policy, create_seq2seq_arc_policy
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_decoder_stack, subsequent_mask


__all__ = []
exporter = export(__all__)


@exporter
class DecoderBase(object):

    def __init__(self, tgt_embedding, **kwargs):
        self.tgt_embedding = tgt_embedding
        self.beam_width = kwargs.get('beam', 1)
        self.best = None
        self.probs = None
        self.preds = None

    def output(self, best, do_probs=True):
        with tf.variable_scope("Output"):
            self.best = tf.identity(best, name='best')
            if self.beam_width > 1 or not do_probs:
                self.probs = tf.no_op(name='probs')
            else:
                self.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), self.preds)

    def predict(self, encoder_outputs, pdrop, **kwargs):
        pass

    def decode(self, encoder_outputs, src_len, tgt_len, pdrop, **kwargs):
        pass


@register_decoder(name='transformer')
class TransformerDecoder(DecoderBase):

    def __init__(self, tgt_embedding, **kwargs):
        super(TransformerDecoder, self).__init__(tgt_embedding, **kwargs)

    @property
    def decoder_type(self):
        return 'transformer'

    def predict(self,
                encoder_outputs,
                src_len,
                pdrop,
                layers=1,
                scope='TransformerDecoder',
                num_heads=4,
                scale=True,
                activation_type='relu',
                d_ff=None,
                **kwargs):
        """self.best is [T, B]"""
        mxlen = kwargs.get('mxlen', 100)
        src_enc = encoder_outputs.output
        B = get_shape_as_list(src_enc)[0]

        if hasattr(encoder_outputs, 'src_mask'):
            src_mask = encoder_outputs.src_mask
        else:
            T = get_shape_as_list(src_enc)[1]
            src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)

        def inner_loop(i, hit_eos, decoded_ids):

            tgt_embed = self.tgt_embedding.encode(decoded_ids)
            T = get_shape_as_list(tgt_embed)[1]
            tgt_mask = subsequent_mask(T)
            scope = 'TransformerDecoder'
            h = transformer_decoder_stack(tgt_embed, src_enc, src_mask, tgt_mask, num_heads, pdrop, scale, layers, activation_type, scope, d_ff)

            vsz = self.tgt_embedding.vsz
            do_weight_tying = bool(kwargs.get('tie_weights', True))  # False
            hsz = get_shape_as_list(h)[-1]
            h = tf.reshape(h, [-1, hsz])
            if do_weight_tying and hsz == self.tgt_embedding.get_dsz():
                with tf.variable_scope(self.tgt_embedding.scope, reuse=True):
                    W = tf.get_variable("W")
                    outputs = tf.matmul(h, W, transpose_b=True, name="logits")
            else:
                vocab_w = tf.get_variable("vocab_w", [hsz, vsz], dtype=tf.float32)
                vocab_b = tf.get_variable("vocab_b", [vsz], dtype=tf.float32)
                outputs = tf.nn.xw_plus_b(h, vocab_w, vocab_b, name="logits")

            preds = tf.reshape(outputs, [B, T, vsz])
            next_id = tf.argmax(preds, axis=-1)[:, -1]
            hit_eos |= tf.equal(next_id, Offsets.EOS)
            next_id = tf.reshape(next_id, [B, 1])

            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i + 1, hit_eos, decoded_ids

        def is_not_finished(i, hit_eos, *_):
            finished = i >= mxlen
            finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        hit_eos = tf.fill([B], False)
        decoded_ids = Offsets.GO * tf.ones([B, 1], dtype=tf.int64)

        _, _, decoded_ids = tf.while_loop(is_not_finished, inner_loop, [tf.constant(0), hit_eos, decoded_ids],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None])

        ])
        self.preds = tf.no_op()
        best = tf.transpose(decoded_ids)
        self.output(best, do_probs=False)

    def decode(self, encoder_outputs,
               src_len,
               tgt_len,
               pdrop,
               layers=1,
               scope='TransformerDecoder',
               num_heads=4,
               scale=True,
               activation_type='relu',
               d_ff=None, **kwargs):
        """self.best is [T, B]"""
        src_enc = encoder_outputs.output
        if hasattr(encoder_outputs, 'src_mask'):
            src_mask = encoder_outputs.src_mask
        else:
            T = get_shape_as_list(src_enc)[1]
            src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)
        tgt_embed = self.tgt_embedding.encode(kwargs.get('tgt'))
        T = get_shape_as_list(tgt_embed)[1]
        tgt_mask = subsequent_mask(T)
        scope = 'TransformerDecoder'
        h = transformer_decoder_stack(tgt_embed, src_enc, src_mask, tgt_mask, num_heads, pdrop, scale, layers, activation_type, scope, d_ff)

        vsz = self.tgt_embedding.vsz
        do_weight_tying = bool(kwargs.get('tie_weights', True))  # False
        hsz = get_shape_as_list(h)[-1]
        if do_weight_tying and hsz == self.tgt_embedding.get_dsz():
            h = tf.reshape(h, [-1, hsz])
            with tf.variable_scope(self.tgt_embedding.scope, reuse=True):
                W = tf.get_variable("W")
                outputs = tf.matmul(h, W, transpose_b=True, name="logits")
        else:
            h = tf.reshape(h, [-1, hsz])
            vocab_w = tf.get_variable("vocab_w", [hsz, vsz], dtype=tf.float32)
            vocab_b = tf.get_variable("vocab_b", [vsz], dtype=tf.float32)
            outputs = tf.nn.xw_plus_b(h, vocab_w, vocab_b, name="logits")
        self.preds = tf.transpose(tf.reshape(outputs, [-1, T, vsz]), [1, 0, 2])
        best = tf.argmax(self.preds, -1)
        self.output(best)

@exporter
class ArcPolicy(object):

    def __init__(self):
        pass

    def connect(self, encoder_outputs, decoder, batch_sz):
        pass


@register_arc_policy(name='no_arc')
class NoArcPolicy(ArcPolicy):

    def __init__(self):
        super(NoArcPolicy, self).__init__()

    def connect(self, encoder_outputs, decoder, batch_sz):
        initial_state = decoder.cell.zero_state(batch_sz*decoder.beam_width, tf.float32)
        return initial_state


class AbstractArcPolicy(ArcPolicy):

    def __init__(self):
        super(AbstractArcPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        pass

    def connect(self, encoder_outputs, decoder, batch_sz):
        final_encoder_state = self.get_state(encoder_outputs)
        final_encoder_state = tf.contrib.seq2seq.tile_batch(final_encoder_state, multiplier=decoder.beam_width)

        initial_state = decoder.cell.zero_state(batch_sz*decoder.beam_width, tf.float32)

        if hasattr(initial_state, 'clone'):
            initial_state = initial_state.clone(cell_state=final_encoder_state)
        else:
            initial_state = final_encoder_state
        return initial_state


@register_arc_policy(name='default')
class TransferLastHiddenPolicy(AbstractArcPolicy):

    def __init__(self):
        super(TransferLastHiddenPolicy, self).__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_decoder(name='vanilla')
class RNNDecoder(DecoderBase):

    def __init__(self, tgt_embedding, **kwargs):
        super(RNNDecoder, self).__init__(tgt_embedding, **kwargs)
        self.hsz = kwargs['hsz']
        self.arc_policy = create_seq2seq_arc_policy(**kwargs)
        self.final_decoder_state = None
        self.do_weight_tying = bool(kwargs.get('tie_weights', False))
        if self.do_weight_tying:
            if self.hsz != self.tgt_embedding.get_dsz():
                raise ValueError("weight tying requires hsz == embedding dsz, \
got {} hsz and {} dsz".format(self.hsz, self.tgt_embedding.get_dsz()))

    @property
    def decoder_type(self):
        return 'vanilla'

    def _create_cell(self, rnn_enc_tensor, src_len, pdrop, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        self.cell = multi_rnn_cell_w_dropout(self.hsz, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())

    def _get_tgt_weights(self):
        Wo = tf.get_variable("Wo", initializer=tf.constant_initializer(self.tgt_embedding.weights,
                                                                       dtype=tf.float32,
                                                                       verify_shape=True),
                             shape=[self.tgt_embedding.vsz, self.tgt_embedding.dsz])
        return Wo

    def predict(self, encoder_outputs, src_len, pdrop, **kwargs):
        """self.best is [T, B, K]"""

        beam_width = kwargs.get('beam', 1)
        mxlen = kwargs.get('mxlen', 100)
        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        Wo = self._get_tgt_weights()
        batch_sz = tf.shape(encoder_outputs.output)[0]
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            proj = dense_layer(self.tgt_embedding.vsz)
            self._create_cell(encoder_outputs.output, src_len, pdrop, **kwargs)
            initial_state = self.arc_policy.connect(encoder_outputs, self, batch_sz)
            # Define a beam-search decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.cell,
                                                           embedding=Wo,
                                                           start_tokens=tf.fill([batch_sz], Offsets.GO),
                                                           end_token=Offsets.EOS,
                                                           initial_state=initial_state,
                                                           beam_width=beam_width,
                                                           output_layer=proj,
                                                           length_penalty_weight=0.0)

            # This creates a "decoder" scope
            final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                      impute_finished=False,
                                                                                      swap_memory=True,
                                                                                      output_time_major=True,
                                                                                      maximum_iterations=mxlen)
            self.final_decoder_state = final_decoder_state
            self.preds = tf.no_op()
            best = final_outputs.predicted_ids
            self.output(best, do_probs=False)

    def decode(self, encoder_outputs, src_len, tgt_len, pdrop, **kwargs):
        """self.best is [T, B]"""
        self.tgt_embedding.x = kwargs.get('tgt', self.tgt_embedding.create_placeholder('tgt'))
        mxlen = kwargs.get('mxlen', 100)

        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        Wo = self._get_tgt_weights()
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            tie_shape = [Wo.get_shape()[-1], Wo.get_shape()[0]]
            if self.do_weight_tying:
                with tf.variable_scope("Share", custom_getter=tie_weight(Wo, tie_shape)):
                    proj = tf.layers.Dense(self.tgt_embedding.vsz, use_bias=False)
            else:
                proj = tf.layers.Dense(self.tgt_embedding.vsz, use_bias=False)

            self._create_cell(encoder_outputs.output, src_len, pdrop, **kwargs)
            batch_sz = tf.shape(encoder_outputs.output)[0]
            initial_state = self.arc_policy.connect(encoder_outputs, self, batch_sz)

            # Two paths depending on training or evaluating (during training)
            # Normal expected inference path is BeamDecoder using .predict()
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, self.tgt_embedding.x), sequence_length=tgt_len)
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(Wo, tf.fill([batch_sz], Offsets.GO), Offsets.EOS)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.cell, helper=training_helper,
                                                      initial_state=initial_state, output_layer=proj)
            training_outputs, self.final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                              impute_finished=True,
                                                                                              swap_memory=True,
                                                                                              output_time_major=True)

            self.preds = training_outputs.rnn_output

            greedy_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.cell,
                                                             helper=greedy_helper,
                                                             initial_state=initial_state, output_layer=proj)
            greedy_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(greedy_decoder,
                                                                     impute_finished=True,
                                                                     swap_memory=True,
                                                                     output_time_major=True,
                                                                     maximum_iterations=mxlen)
            best = greedy_outputs.sample_id
            self.output(best)


@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):
    def __init__(self, tgt_embedding, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embedding, **kwargs)
        self.attn_type = kwargs.get('attn_type', 'bahdanau').lower()

    @property
    def decoder_type(self):
        return 'default'

    @property
    def attn_type(self):
        return self._attn_type

    @attn_type.setter
    def attn_type(self, type):
        self._attn_type = type

    def _create_cell(self, rnn_enc_tensor, src_len, pdrop, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        cell = multi_rnn_cell_w_dropout(self.hsz, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())
        if self.beam_width > 1:
            # Expand the encoded tensor for all beam entries
            rnn_enc_tensor = tf.contrib.seq2seq.tile_batch(rnn_enc_tensor, multiplier=self.beam_width)
            src_len = tf.contrib.seq2seq.tile_batch(src_len, multiplier=self.beam_width)
        GlobalAttention = tfcontrib_seq2seq.LuongAttention if self.attn_type == 'luong' else tfcontrib_seq2seq.BahdanauAttention
        attn_mech = GlobalAttention(self.hsz, rnn_enc_tensor, src_len)
        self.cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
