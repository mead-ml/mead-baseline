from baseline.tf.tfy import *


def transformer_encoder(x, src_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
    d_model = get_shape_as_list(x)[-1]
    return TransformerEncoder(d_model, num_heads, pdrop, scale, activation_type, d_ff, name=scope)(x, TRAIN_FLAG(),
                                                                                                   src_mask)


def transformer_decoder(src, tgt, src_mask, tgt_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):

    d_model = get_shape_as_list(tgt)[-1]
    return TransformerDecoder(d_model, num_heads, pdrop, scale, activation_type, d_ff)((src, tgt), TRAIN_FLAG(), (src_mask, tgt_mask))


def transformer_encoder_stack(x, src_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerEncoder', d_ff=None):
    d_model = get_shape_as_list(x)[-1]

    return TransformerEncoderStack(d_model, num_heads, pdrop, scale, layers, activation_type, d_ff=d_ff, name=scope)(x, TRAIN_FLAG(), src_mask)


def transformer_decoder_stack(src, tgt, src_mask, tgt_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerDecoder', d_ff=None):
    d_model = get_shape_as_list(tgt)[-1]
    return TransformerDecoderStack(d_model, num_heads, pdrop, scale, layers, activation_type, d_ff=d_ff, name=scope)((src, tgt), TRAIN_FLAG(), (src_mask, tgt_mask))


def subsequent_mask(size):
    b = tf.matrix_band_part(tf.ones([size, size]), -1, 0)
    m = tf.reshape(b, [1, 1, size, size])
    return m
