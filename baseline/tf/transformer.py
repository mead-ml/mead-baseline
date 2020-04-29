from baseline.tf.tfy import *
from eight_mile.tf.layers import subsequent_mask


def transformer_encoder(x, src_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
    d_model = get_shape_as_list(x)[-1]
    return TransformerEncoder(d_model, num_heads, pdrop, scale, activation_type, d_ff, name=scope)((x, src_mask))


def transformer_decoder(src, tgt, src_mask, tgt_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
    d_model = get_shape_as_list(tgt)[-1]
    return TransformerDecoder(d_model, num_heads, pdrop, scale, activation_type, d_ff)((src, tgt, src_mask, tgt_mask))


def transformer_encoder_stack(x, src_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerEncoder', d_ff=None):
    d_model = get_shape_as_list(x)[-1]
    return TransformerEncoderStack(d_model, num_heads, pdrop, scale, layers, activation_type, d_ff=d_ff, name=scope)((x, src_mask))


def transformer_decoder_stack(tgt, src, src_mask, tgt_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerDecoder', d_ff=None):
    d_model = get_shape_as_list(tgt)[-1]
    return TransformerDecoderStack(d_model, num_heads, pdrop, scale, layers, activation_type, d_ff=d_ff, name=scope)((tgt, src, src_mask, tgt_mask))
