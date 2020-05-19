from baseline.tf.tfy import *
from baseline.tf.embeddings import *
from baseline.tf.transformer import transformer_encoder_stack
from baseline.utils import exporter, MAGIC_VARS
from baseline.model import register_encoder
from collections import namedtuple

__all__ = []
export = exporter(__all__)



RNNEncoderOutput = namedtuple("RNNEncoderOutput", ("output", "hidden", "src_mask"))


def _make_src_mask(output, lengths):
    T = tf.shape(output)[1]
    src_mask = tf.cast(tf.sequence_mask(lengths, T), dtype=tf.uint8)
    return src_mask


@export
class EncoderBase(tf.keras.layers.Layer):

    def __init__(self, name='encoder', **kwargs):
        super().__init__(name=name)

    def call(self, inputs, **kwargs):
        pass


@register_encoder(name='default')
class RNNEncoder(EncoderBase):

    def __init__(self, name='encoder', pdrop=0.1, hsz=650, rnntype='blstm', layers=1, vdrop=False, scope='RNNEncoder', residual=False, create_src_mask=True, **kwargs):
        super().__init__(name=name)
        Encoder = BiLSTMEncoderAllLegacy if rnntype == 'blstm' else LSTMEncoderAllLegacy
        self.rnn = Encoder(None, hsz, layers, pdrop, vdrop, name=scope)
        self.residual = residual
        self.src_mask_fn = _make_src_mask if create_src_mask is True else lambda x, y: None

    @property
    def encoder_type(self):
        return 'default'

    def call(self, inputs, **kwargs):
        embed_in, src_len = inputs

        # This comes out as a sequence T of (B, D)
        output, hidden = self.rnn((embed_in, src_len))
        output = output + embed_in if self.residual else output
        return RNNEncoderOutput(output=output, hidden=hidden, src_mask=self.src_mask_fn(output, src_len))


TransformerEncoderOutput = namedtuple("TransformerEncoderOutput", ("output", "src_mask"))


@register_encoder(name='transformer')
class TransformerEncoder(EncoderBase):

    def __init__(self, pdrop=0.1, hsz=650, num_heads=4, layers=1, scale=True, activation_type='relu', name="encode", d_ff=None, scope="TransformerEncoder", **kwargs):
        super().__init__(name=name)
        self.encoder = TransformerEncoderStack(num_heads, hsz, pdrop, scale, layers, activation_type, d_ff, name=scope)

    @property
    def encoder_type(self):
        return 'transformer'

    def call(self, inputs, **kwargs):
        embed_in, src_len = inputs
        T = get_shape_as_list(embed_in)[1]
        src_mask = tf.sequence_mask(src_len, T, dtype=tf.float32)
        shp = get_shape_as_list(src_mask)
        new_shp = [shp[0]] + [1, 1] + shp[1:]
        src_mask = tf.reshape(src_mask, new_shp)
        encoder_output = self.encoder((embed_in, src_mask))
        # This comes out as a sequence T of (B, D)
        return TransformerEncoderOutput(output=encoder_output, src_mask=src_mask)

