from baseline.tf.tfy import *
import tensorflow.contrib.seq2seq as tfcontrib_seq2seq
from baseline.utils import ls_props, read_json, Offsets
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_decoder_stack, subsequent_mask


class DecoderBase(object):

    def __init__(self, tgt_embedding, **kwargs):
        self.tgt_embedding = tgt_embedding
        self.beam_width = kwargs.get('beam', 1)
        self.best = None
        self.probs = None

    def output(self, best, **kwargs):
        with tf.variable_scope("Output"):
            self.best = tf.identity(best, name='best')
            if self.beam_width > 1:
                self.probs = tf.no_op(name='probs')
            else:
                self.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), self.preds)

    def predict(self, encoder_outputs, pkeep, **kwargs):
        pass

    def decode(self, encoder_outputs, src_len, tgt_len, pkeep, **kwargs):
        pass


class TransformerDecoder(DecoderBase):

    def __init__(self, tgt_embedding, **kwargs):
        super(TransformerDecoder, self).__init__(tgt_embedding, **kwargs)

    def predict(self,
                encoder_outputs,
                pkeep,
                layers=1,
                scope='TransformerDecoder',
                num_heads=4,
                scale=True,
                activation_type='relu',
                d_ff=None,
                **kwargs):
        raise Exception('Implement me!')

    def decode(self, encoder_outputs,
               src_len,
               tgt_len,
               pkeep,
               layers=1,
               scope='TransformerDecoder',
               num_heads=4,
               scale=True,
               activation_type='relu',
               d_ff=None, **kwargs):
#        self.tgt_embedding.x = self.tgt_embedding.create_placeholder(self.tgt_embedding.name)
        src_enc = encoder_outputs.output
        src_mask = encoder_outputs.src_mask

        tgt_embed = self.tgt_embedding.encode()
        T = get_shape_as_list(tgt_embed)[1]
        tgt_mask = subsequent_mask(T)
        scope = 'TransformerDecoder'
        h = transformer_decoder_stack(src_enc, tgt_embed, src_mask, tgt_mask, num_heads, pkeep, scale, layers, activation_type, scope, d_ff)

        vsz = self.tgt_embedding.vsz
        do_weight_tying = bool(kwargs.get('tie_weights', True))  # False
        hsz = get_shape_as_list(h)[-1]
        if do_weight_tying and hsz == self.tgt_embedding.get_dsz():
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

class RNNDecoder(DecoderBase):

    def __init__(self, tgt_embedding, **kwargs):
        super(RNNDecoder, self).__init__(tgt_embedding, **kwargs)
        self.hsz = kwargs['hsz']

    def _create_cell(self, rnn_enc_tensor, src_len, pkeep, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        cell = multi_rnn_cell_w_dropout(self.hsz, pkeep, rnntype, layers, variational=vdrop)
        return cell

    def bridge(self, final_encoder_state):
        return final_encoder_state

    def _get_tgt_weights(self):
        Wo = tf.get_variable("Wo", initializer=tf.constant_initializer(self.tgt_embedding.weights,
                                                                       dtype=tf.float32,
                                                                       verify_shape=True),
                             shape=[self.tgt_embedding.vsz, self.tgt_embedding.dsz])
        return Wo

    def predict(self, encoder_outputs, src_len, pkeep, **kwargs):

        beam_width = kwargs.get('beam', 1)
        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        Wo = self._get_tgt_weights()
        batch_sz = tf.shape(encoder_outputs.output)[0]
        final_encoder_state = encoder_outputs.hidden
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            proj = dense_layer(self.tgt_embedding.vsz)
            self._create_cell(encoder_outputs.output, src_len, pkeep, **kwargs)
            final_encoder_state = tf.contrib.seq2seq.tile_batch(final_encoder_state, multiplier=self.beam_width)
            initial_state = self.bridge(final_encoder_state)
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
                                                                                      output_time_major=True)
            self.preds = tf.no_op()
            best = final_outputs.predicted_ids
            self.output(best)

    def decode(self, encoder_outputs, src_len, tgt_len, pkeep, **kwargs):
        self.tgt_embedding.x = self.tgt_embedding.create_placeholder(self.tgt_embedding.name)

        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        Wo = self._get_tgt_weights()
        final_encoder_state = encoder_outputs.hidden
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            proj = dense_layer(self.tgt_embedding.vsz)
            self._create_cell(encoder_outputs.output, src_len, pkeep, **kwargs)
            initial_state = self.bridge(final_encoder_state, tf.shape(encoder_outputs.output)[0])
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, self.tgt_embedding.x), sequence_length=tgt_len)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.cell, helper=helper, initial_state=initial_state, output_layer=proj)
            final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                      impute_finished=True,
                                                                                      swap_memory=True,
                                                                                      output_time_major=True)
            self.preds = final_outputs.rnn_output
            best = final_outputs.sample_id
            self.output(best)


class RNNDecoderWithAttn(RNNDecoder):
    def __init__(self, tgt_embedding, **kwargs):
        super(RNNDecoderWithAttn, self).__init__(tgt_embedding, **kwargs)
        self.attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        self.arc_state = kwargs.get('arc_state', False)

    def _create_cell(self, rnn_enc_tensor, src_len, pkeep, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        cell = multi_rnn_cell_w_dropout(self.hsz, pkeep, rnntype, layers, variational=vdrop)
        if self.beam_width > 1:
            # Expand the encoded tensor for all beam entries
            rnn_enc_tensor = tf.contrib.seq2seq.tile_batch(rnn_enc_tensor, multiplier=self.beam_width)
            src_len = tf.contrib.seq2seq.tile_batch(src_len, multiplier=self.beam_width)
        GlobalAttention = tfcontrib_seq2seq.LuongAttention if self.attn_type == 'luong' else tfcontrib_seq2seq.BahdanauAttention
        attn_mech = GlobalAttention(self.hsz, rnn_enc_tensor, src_len)
        self.cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
        return self.cell

    def bridge(self, final_encoder_state, batch_sz):
        initial_state = self.cell.zero_state(batch_sz*self.beam_width, tf.float32)
        if self.arc_state is True:
            initial_state = initial_state.clone(cell_state=final_encoder_state)
        return initial_state
