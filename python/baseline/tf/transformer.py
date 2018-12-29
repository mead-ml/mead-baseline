import tensorflow as tf
from baseline.tf.tfy import tf_activation, get_shape_as_list, layer_norm, time_distributed_projection
from baseline.tf.layers import TRAIN_FLAG, FFN, split_heads, combine_heads, TransformerEncoder



def self_attention_qkv(x, d_model, scope='self_attn'):
    return low_order_projection_qkv(x, x, x, d_model, scope=scope)
    ###c = time_distributed_projection(x, name='qkv_conv', filters=d_model*3)
    ### Split into 3 pieces along 2nd axis
    ###q, k, v = tf.split(c, 3, 2)
    return q, k, v


def low_order_projection_qkv(q_in, k_in, v_in, d_model, scope='low_order_proj'):
    with tf.variable_scope(scope):
        q = time_distributed_projection(q_in, name='q_conv', filters=d_model)
        k = time_distributed_projection(k_in, name='k_conv', filters=d_model)
        v = time_distributed_projection(v_in, name='v_conv', filters=d_model)
        ###kv = time_distributed_projection(x, name='kv_conv', filters=d_model*2)
    ###k, v = tf.split(kv, 2, 2)
    return q, k, v


def multi_headed_attention(q, k, v, scope, d_model, num_heads, pdrop, scale=False, mask=None):
    assert d_model % num_heads == 0
    with tf.variable_scope(scope):
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        a = dot_product_attention(q, k, v, pdrop, scale=scale, mask=mask)
        a = combine_heads(a)
        a = time_distributed_projection(a, name='attn_conv', filters=d_model)
        return a


def ffn(x, scope, pdrop, d_ff=None, activation_type='relu'):
    d_model = get_shape_as_list(x)[-1]
    return FFN(d_model, pdrop, activation_type, d_ff, name=scope)(x, training=TRAIN_FLAG())




#def transformer_encoder(x, src_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
#
#    with tf.variable_scope(scope):
#        d_model = get_shape_as_list(x)[-1]
#        if d_ff is None:
#            d_ff = 4*d_model
#        x = layer_norm(x, 'ln_1')
#        q, k, v = self_attention_qkv(x, d_model)
#        a = multi_headed_attention(q, k, v, 'attn', d_model, num_heads, pdrop, scale=scale, mask=src_mask)
#        x = x + tf.layers.dropout(a, pdrop, training=TRAIN_FLAG())
#        x = layer_norm(x, 'ln_2')
#        m = ffn(x, 'ffn', pdrop, d_ff=d_ff, activation_type=activation_type)
#        h = x + tf.layers.dropout(m, pdrop, training=TRAIN_FLAG())
#        return h

def transformer_encoder(x, src_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
    d_model = get_shape_as_list(x)[-1]
    return TransformerEncoder(d_model, num_heads, pdrop, scale, activation_type, d_ff, name=scope)(x, TRAIN_FLAG(),
                                                                                                   src_mask)


def transformer_decoder(src, tgt, src_mask, tgt_mask, scope, num_heads, pdrop, scale=True, activation_type='relu', d_ff=None):
    with tf.variable_scope(scope):
        d_model = get_shape_as_list(tgt)[-1]
        if d_ff is None:
            d_ff = 4*d_model

        tgt = layer_norm(tgt, 'ln_1')

        q, k, v = self_attention_qkv(tgt, d_model)
        self_attn = multi_headed_attention(q, k, v, 'self_attn', d_model, num_heads, pdrop, scale=scale, mask=tgt_mask)
        tgt = tgt + tf.layers.dropout(self_attn, pdrop, training=TRAIN_FLAG())
        tgt = layer_norm(tgt, 'ln_2')

        q, k, v = low_order_projection_qkv(tgt, src, src, d_model)
        # Mask at zeros???
        src_attn = multi_headed_attention(q, k, v, "dual_attn", d_model, num_heads, pdrop, scale=scale, mask=src_mask)
        tgt = tgt + tf.layers.dropout(src_attn, pdrop, training=TRAIN_FLAG())

        tgt = layer_norm(tgt, 'ln_3')
        m = ffn(tgt, 'ffn', pdrop, d_ff=d_ff, activation_type=activation_type)
        h = tgt + tf.layers.dropout(m, pdrop, training=TRAIN_FLAG())
        return h


def transformer_encoder_stack(x, src_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerEncoder', d_ff=None):
    with tf.variable_scope(scope):
        for i in range(layers):
            x = transformer_encoder(x, src_mask, 'encoder-{}'.format(i), num_heads, pdrop, scale, activation_type, d_ff)
    return layer_norm(x, 'ln_out')


def transformer_decoder_stack(src, tgt, src_mask, tgt_mask, num_heads, pdrop, scale=True, layers=1, activation_type='relu', scope='TransformerEncoder', d_ff=None):
    with tf.variable_scope(scope):
        for i in range(layers):
            x = transformer_decoder(src, tgt, src_mask, tgt_mask, 'decoder-{}'.format(i), num_heads, pdrop, scale, activation_type, d_ff)
    return layer_norm(x, 'ln_out')


def subsequent_mask(size):
    b = tf.matrix_band_part(tf.ones([size, size]), -1, 0)
    m = tf.reshape(b, [1, 1, size, size])
    return m


def dot_product_attention(query, key, value, pdrop=0.0, mask=None, scale=True):
    w = tf.matmul(query, key, transpose_b=True)

    if scale:
        w *= tf.rsqrt(tf.to_float(tf.shape(query)[2]))

    if mask is not None:
        w = w * mask + -1e9 * (1 - mask)

    weights = tf.nn.softmax(w, name="attention_weights")
    weights = tf.layers.dropout(weights, pdrop, training=TRAIN_FLAG())
    return tf.matmul(weights, value)

