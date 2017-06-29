import tensorflow as tf
import numpy as np
from distutils.version import LooseVersion
from tensorflow.python.ops import rnn_cell_impl
TF_GTE_11 = LooseVersion(tf.__version__) >= LooseVersion('0.1.1')
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial

def tensor2seq(tensor):
    return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))


def seq2tensor(sequence):
    return tf.transpose(tf.stack(sequence), perm=[1, 0, 2])


# Method for seq2seq w/ attention using TF's library
def legacy_attn_rnn_seq2seq(encoder_inputs,
                            decoder_inputs,
                            cell,
                            num_heads=1,
                            dtype=tf.float32,
                            scope=None):
    with tf.variable_scope(scope or "attention_rnn_seq2seq"):
        encoder_outputs, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=dtype)
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(values=top_states, axis=1)
    
    return tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs,
                                                       enc_state,
                                                       attention_states,
                                                       cell,
                                                       num_heads=num_heads)


def dense_layer(output_layer_depth):
    output_layer = layers_core.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
    return output_layer
    

def new_rnn_cell(hsz, rnntype, st=None):
    if st is not None:
        return tf.contrib.rnn.BasicLSTMCell(hsz, state_is_tuple=st) if rnntype == 'lstm' else tf.contrib.rnn.GRUCell(hsz)
    return tf.contrib.rnn.LSTMCell(hsz) if rnntype == 'lstm' else tf.contrib.rnn.GRUCell(hsz)


def new_multi_rnn_cell(hsz, name, num_layers):
    return tf.contrib.rnn.MultiRNNCell([new_rnn_cell(hsz, name) for _ in range(num_layers)], state_is_tuple=True)


def show_examples_tf(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.shape[0])
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    for src_len_i,src_i,tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i, reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = np.zeros((1, mxlen))
        src_i = src_i[np.newaxis,:]
        src_len_i = np.array([src_len_i])
        next_value = GO
        for j in range(mxlen):
            dst_i[0, j] = next_value
            tgt_len_i = np.array([j+1])
            output = model.step(src_i, src_len_i, dst_i, tgt_len_i)[j]
            if sample is False:
                next_value = np.argmax(output)
            else:
                # This is going to zero out low prob. events so they are not
                # sampled from
                next_value = beam_multinomial(prob_clip, output)

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')
