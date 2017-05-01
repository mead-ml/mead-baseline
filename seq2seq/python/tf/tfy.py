import tensorflow as tf
import numpy as np
from distutils.version import LooseVersion
from tensorflow.python.ops import rnn_cell_impl
TF_GTE_11 = LooseVersion(tf.__version__) >= LooseVersion('0.1.1')
import math
from utils import *
from tensorflow.python.layers import core as layers_core

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

def print_batch(best, rlut2):
    batchsz = best.shape[1]
    print(batchsz)
    T = min(best.shape[0], 20)
    for b in range(batchsz):
        ll = []
        for i in range(best.shape[0]):
            v = rlut2[best[i][b]]
            if v == '<EOS>':
                break
            if v != '<PADDING>':
                ll += [rlut2[best[i][b]]]
        print(ll)
