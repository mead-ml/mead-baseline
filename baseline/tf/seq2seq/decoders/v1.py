from baseline.tf.tfy import *
from baseline.utils import ls_props, read_json, Offsets, exporter
from baseline.model import register_decoder, register_arc_policy, create_seq2seq_arc_policy
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_decoder_stack, subsequent_mask


__all__ = []
export = exporter(__all__)




import tensorflow.contrib.seq2seq as tfcontrib_seq2seq

@export
class DecoderBase(tf.keras.layers.Layer):

    def __init__(self, tgt_embedding, name='decoder', **kwargs):
        super().__init__(name=name)
        self.tgt_embedding = tgt_embedding
        self.beam_width = kwargs.get('beam', 1)
        self.best = None
        self.probs = None
        self.preds = None
        self.mxlen = kwargs.get('mxlen', 100)

    def output(self, best, do_probs=True):
        with tf.variable_scope("Output"):
            self.best = tf.identity(best, name='best')
            if self.beam_width > 1 or not do_probs:
                self.probs = tf.no_op(name='probs')
            else:
                self.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), self.preds)

    def call(self, inputs, **kwargs):
        if kwargs.get('predict', False):
            return self.predict(inputs)
        return self.decode(inputs)

    def predict(self, inputs, **kwargs):
        pass

    def decode(self, inputs, **kwargs):
        pass


@register_decoder(name='transformer')
class TransformerDecoder(DecoderBase):

    def __init__(self, tgt_embedding, pdrop=0.1, layers=1, name='decode', num_heads=4, scale=True, activation_type='relu', d_ff=None, scope='TransformerDecoder', **kwargs):
        super().__init__(tgt_embedding, name=name, **kwargs)
        # In predict mode the placeholder for the tgt embedding isn't created so the weights in the tgt embedding object
        # is called `tgt/LUT/weights` because there isn't a placeholder called `tgt`. In decode where that placeholder
        # exists the weights are called `tgt_1/LUT/weights`
        #if kwargs.get('predict', False):
        #    tf.no_op(name=f"{name}/{self.tgt_embedding.name}")
        dsz = self.tgt_embedding.get_dsz()
        vsz = self.tgt_embedding.get_vsz()
        self.decoder = TransformerDecoderStack(dsz, num_heads, pdrop, scale, layers, activation_type, d_ff, name=scope)
        self.do_weight_tying = bool(kwargs.get('tie_weights', True))
        if self.do_weight_tying:
            self.proj = WeightTieDense(self.tgt_embedding)
        else:
            self.proj = tf.keras.layers.Dense(vsz, use_bias=False)

    @property
    def decoder_type(self):
        return 'transformer'

    def predict(self, inputs, **kwargs):
        """self.best is [T, B]"""
        encoder_outputs, src_len = inputs
        src_enc = encoder_outputs.output
        B = get_shape_as_list(src_enc)[0]

        if hasattr(encoder_outputs, 'src_mask'):
            src_mask = encoder_outputs.src_mask
        else:
            T = get_shape_as_list(src_enc)[1]
            src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)

        def inner_loop(i, hit_eos, decoded_ids):

            tgt_embed = self.tgt_embedding(decoded_ids)
            T = get_shape_as_list(tgt_embed)[1]
            tgt_mask = subsequent_mask(T)
            h = self.decoder((tgt_embed, src_enc, src_mask, tgt_mask))
            # hsz = get_shape_as_list(h)[-1]
            # h = tf.reshape(h, [-1, hsz])
            outputs = self.proj(h)

            preds = tf.reshape(outputs, [B, T, -1])
            next_id = tf.argmax(preds, axis=-1)[:, -1]
            hit_eos |= tf.equal(next_id, Offsets.EOS)
            next_id = tf.reshape(next_id, [B, 1])

            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return i + 1, hit_eos, decoded_ids

        def is_not_finished(i, hit_eos, *_):
            finished = i >= self.mxlen
            finished |= tf.reduce_all(hit_eos)
            return tf.logical_not(finished)

        hit_eos = tf.fill([B], False)
        decoded_ids = Offsets.GO * tf.ones([B, 1], dtype=tf.int64)
        # Call the inner loop once so that the tgt weights aren't prefixed with `while`
        i, hit_eos, decoded_ids = inner_loop(tf.constant(0), hit_eos, decoded_ids)

        _, _, decoded_ids = tf.while_loop(is_not_finished, inner_loop, [i, hit_eos, decoded_ids],
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None])

        ])
        self.preds = tf.no_op()
        best = tf.transpose(decoded_ids)
        self.output(best, do_probs=False)
        return self.best

    def decode(self, inputs):
        encoder_outputs, tgt, src_len, tgt_len = inputs
        tgt_embed = self.tgt_embedding(tgt)
        #if not tgt:
        #    tgt = self.tgt_embedding.create_placeholder(self.tgt_embedding.name)
        src_enc = encoder_outputs.output

        #tgt_embed = self.tgt_embedding.encode(tgt)
        shape = get_shape_as_list(tgt_embed)
        B = shape[0]
        T = shape[1]

        if hasattr(encoder_outputs, 'src_mask'):
            src_mask = encoder_outputs.src_mask
        else:
            src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)

        tgt_mask = subsequent_mask(T)
        h = self.decoder((tgt_embed, src_enc, src_mask, tgt_mask))
        outputs = self.proj(h)

        self.preds = tf.transpose(tf.reshape(outputs, [B, T, -1]), [1, 0, 2])
        best = tf.argmax(self.preds, -1)
        self.output(best)
        return self.best


@export
class ArcPolicy(object):

    def __init__(self):
        pass

    def connect(self, encoder_outputs, decoder, batch_sz):
        pass


@register_arc_policy(name='no_arc')
class NoArcPolicy(ArcPolicy):

    def __init__(self):
        super().__init__()

    def connect(self, encoder_outputs, decoder, batch_sz):
        initial_state = decoder.cell.zero_state(batch_sz*decoder.beam_width, tf.float32)
        return initial_state


class AbstractArcPolicy(ArcPolicy):

    def __init__(self):
        super().__init__()

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
        super().__init__()

    def get_state(self, encoder_outputs):
        return encoder_outputs.hidden


@register_decoder(name='vanilla')
class RNNDecoder(DecoderBase):

    def __init__(self, tgt_embedding, hsz, pdrop, rnntype='lstm', layers=1, vdrop=False, name='encoder', scope='RNNDecoder', **kwargs):
        super().__init__(tgt_embedding, **kwargs)
        self.hsz = hsz
        self.pdrop = pdrop
        self.rnntype = rnntype
        self.layers = layers
        self.vdrop = vdrop
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

    def _create_cell(self, rnn_enc_tensor, src_len, hsz, pdrop, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        return multi_rnn_cell_w_dropout(hsz, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())

    def _get_tgt_weights(self):
        Wo = tf.get_variable("Wo", initializer=tf.constant_initializer(self.tgt_embedding._weights,
                                                                       dtype=tf.float32,
                                                                       verify_shape=True),
                             shape=[self.tgt_embedding.get_vsz(), self.tgt_embedding.get_dsz()])
        return Wo

    def predict(self, inputs, **kwargs):
        """self.best is [T, B, K]"""
        encoder_outputs, src_len = inputs

        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        # This is a huge hack where we are getting and wrapping the weight from the tgt embedding
        # which never actually gets used inside the embeddings object to create a tensor
        Wo = self._get_tgt_weights()
        batch_sz = tf.shape(encoder_outputs.output)[0]
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            # We just create a normal dense layer, the checkpoint will populate it with the right value
            proj = dense_layer(self.tgt_embedding.get_vsz())
            self.cell = self._create_cell(encoder_outputs.output, src_len, self.hsz, self.pdrop, self.rnntype, self.layers, self.vdrop, **kwargs)
            initial_state = self.arc_policy.connect(encoder_outputs, self, batch_sz)
            # Define a beam-search decoder
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.cell,
                embedding=Wo,
                start_tokens=tf.fill([batch_sz], Offsets.GO),
                end_token=Offsets.EOS,
                initial_state=initial_state,
                beam_width=self.beam_width,
                output_layer=proj,
                length_penalty_weight=0.0
            )

            # This creates a "decoder" scope
            final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                impute_finished=False,
                swap_memory=True,
                output_time_major=True,
                maximum_iterations=self.mxlen
            )
            self.final_decoder_state = final_decoder_state
            self.preds = tf.no_op()
            best = final_outputs.predicted_ids
            self.output(best, do_probs=False)
            return self.best

    def decode(self, inputs, **kwargs):
        """self.best is [T, B]"""
        encoder_outputs, tgt, src_len, tgt_len = inputs
        self.tgt_embedding.x = tgt if tgt is not None else self.tgt_embedding.create_placeholder(self.tgt_embedding.name)

        # dynamic_decode creates a scope "decoder" and it pushes operations underneath.
        # which makes it really hard to get the same objects between train and test
        # In an ideal world, TF would just let us using tgt_embedding.encode as a function pointer
        # This works fine for training, but then at decode time its not quite in the right place scope-wise
        # So instead, for now, we never call .encode() and instead we create our own operator
        # This is a huge hack where we are getting and wrapping the weight from the tgt embedding
        # which never actually gets used inside the embeddings object to create a tensor
        Wo = self._get_tgt_weights()
        with tf.variable_scope("dec", reuse=tf.AUTO_REUSE):
            tie_shape = [Wo.get_shape()[-1], Wo.get_shape()[0]]
            self.cell = self._create_cell(encoder_outputs.output, src_len, self.hsz, self.pdrop, self.rnntype, self.layers, self.vdrop, **kwargs)
            if self.do_weight_tying:
                with tf.variable_scope("Share", custom_getter=tie_weight(Wo, tie_shape)):
                    proj = tf.layers.Dense(self.tgt_embedding.get_vsz(), use_bias=False)
            else:
                proj = tf.layers.Dense(self.tgt_embedding.get_vsz(), use_bias=False)

            batch_sz = tf.shape(encoder_outputs.output)[0]
            initial_state = self.arc_policy.connect(encoder_outputs, self, batch_sz)

            # Two paths depending on training or evaluating (during training)
            # Normal expected inference path is BeamDecoder using .predict()
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(Wo, self.tgt_embedding.x),
                sequence_length=tgt_len
            )
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.cell,
                helper=training_helper,
                initial_state=initial_state,
                output_layer=proj
            )
            training_outputs, self.final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                training_decoder,
                impute_finished=True,
                swap_memory=True,
                output_time_major=True
            )

            self.preds = training_outputs.rnn_output

            # This is used to do greedy decoding during evaluation on the dev set
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                Wo,
                tf.fill([batch_sz], Offsets.GO),
                Offsets.EOS
            )
            greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.cell,
                helper=greedy_helper,
                initial_state=initial_state,
                output_layer=proj
            )
            greedy_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                greedy_decoder,
                impute_finished=True,
                swap_memory=True,
                output_time_major=True,
                maximum_iterations=self.mxlen
            )
            best = greedy_outputs.sample_id
            self.output(best)
            return self.best


@register_decoder(name='default')
class RNNDecoderWithAttn(RNNDecoder):
    def __init__(self, tgt_embedding, **kwargs):
        self.attn_type = kwargs.get('attn_type', 'bahdanau').lower()
        super().__init__(tgt_embedding, **kwargs)

    @property
    def decoder_type(self):
        return 'default'

    @property
    def attn_type(self):
        return self._attn_type

    @attn_type.setter
    def attn_type(self, type):
        self._attn_type = type

    def _create_cell(self, rnn_enc_tensor, src_len, hsz, pdrop, rnntype='lstm', layers=1, vdrop=False, **kwargs):
        cell = multi_rnn_cell_w_dropout(hsz, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())
        if self.beam_width > 1:
            # Expand the encoded tensor for all beam entries
            rnn_enc_tensor = tf.contrib.seq2seq.tile_batch(rnn_enc_tensor, multiplier=self.beam_width)
            src_len = tf.contrib.seq2seq.tile_batch(src_len, multiplier=self.beam_width)
        GlobalAttention = tfcontrib_seq2seq.LuongAttention if self.attn_type == 'luong' else tfcontrib_seq2seq.BahdanauAttention
        attn_mech = GlobalAttention(hsz, rnn_enc_tensor, src_len)
        return tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
