from baseline.tf.tfy import *
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_encoder_stack
from baseline.utils import export
from baseline.model import register_encoder
from collections import namedtuple

__all__ = []
exporter = export(__all__)


@exporter
class EncoderBase(object):

    def __init__(self):
        pass

    def encode(self, embed_in, src_len, pdrop, **kwargs):
        pass


RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden"))


@register_encoder(name='default')
class RNNEncoder(EncoderBase):

    def __init__(self, **kwargs):
        super(RNNEncoder, self).__init__()

    @property
    def encoder_type(self):
        return 'default'

    def encode(self, embed_in, src_len, pdrop, hsz=650, rnntype='blstm', layers=1, vdrop=False, **kwargs):

        if rnntype == 'blstm':
            rnn_fwd_cell = multi_rnn_cell_w_dropout(hsz//2, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())
            rnn_bwd_cell = multi_rnn_cell_w_dropout(hsz//2, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())
            rnn_enc_tensor, (fw_final_state, bw_final_state) = tf.nn.bidirectional_dynamic_rnn(rnn_fwd_cell, rnn_bwd_cell,
                                                                                               embed_in,
                                                                                               scope='brnn_enc',
                                                                                               sequence_length=src_len,
                                                                                               dtype=tf.float32)

            rnn_enc_tensor = tf.concat(rnn_enc_tensor, -1)
            encoder_state = []
            for i in range(layers):
                h = tf.concat([fw_final_state[i].h, bw_final_state[i].h], -1)
                c = tf.concat([fw_final_state[i].c, bw_final_state[i].c], -1)
                encoder_state.append(tf.contrib.rnn.LSTMStateTuple(h=h, c=c))
            encoder_state = tuple(encoder_state)
        else:

            rnn_enc_cell = multi_rnn_cell_w_dropout(hsz, pdrop, rnntype, layers, variational=vdrop, training=TRAIN_FLAG())
            rnn_enc_tensor, encoder_state = tf.nn.dynamic_rnn(rnn_enc_cell, embed_in,
                                                              scope='rnn_enc',
                                                              sequence_length=src_len,
                                                              dtype=tf.float32)

        # This comes out as a sequence T of (B, D)
        return RNNEncoderOutput(output=rnn_enc_tensor, hidden=encoder_state)


TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))


@register_encoder(name='transformer')
class TransformerEncoder(EncoderBase):

    def __init__(self, **kwargs):
        super(TransformerEncoder, self).__init__()

    @property
    def encoder_type(self):
        return 'transformer'

    def encode(self,
               embed_in,
               src_len,
               pdrop,
               hsz=650,
               num_heads=4,
               layers=1,
               scale=True,
               activation_type='relu',
               scope='TransformerEncoder',
               d_ff=None,
               **kwargs):
        T = get_shape_as_list(embed_in)[1]
        src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)
        shp = get_shape_as_list(src_mask)
        new_shp = [shp[0]] + [1, 1] + shp[1:]
        src_mask = tf.reshape(src_mask, new_shp)
        encoder_output = transformer_encoder_stack(embed_in, src_mask, num_heads,
                                                   pdrop, scale, layers, activation_type, scope, d_ff)
        # This comes out as a sequence T of (B, D)
        return TransformerEncoderOutput(output=encoder_output, src_mask=src_mask)

