import tensorflow as tf
import numpy as np

# Method for seq2seq w/ attention using TF's library
def attn_rnn_seq2seq(encoder_inputs,
                     decoder_inputs,
                     cell,
                     num_heads=1,
                     dtype=tf.float32,
                     scope=None):
    with tf.variable_scope(scope or "attention_rnn_seq2seq"):
        encoder_outputs, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=dtype)
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)
    
    return tf.nn.seq2seq.attention_decoder(decoder_inputs,
                                           enc_state,
                                           attention_states,
                                           cell,
                                           num_heads=num_heads)

def tensorToSeq(tensor):
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def seqToTensor(sequence):
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

def revlut(lut):
    return {v: k for k, v in lut.items()}

def lookupSent(rlut, seq, reverse=False):
    s = seq[::-1] if reverse else seq
    return ' '.join([rlut[idx] if rlut[idx] != '<PADDING>' else '' for idx in s])


# Get a sparse index (dictionary) of top values
# Note: mutates input for efficiency
def topk(k, probs):

    lut = {}
    i = 0

    while i < k:
        idx = np.argmax(probs)
        lut[idx] = probs[idx]
        probs[idx] = 0
        i += 1
    return lut

#  Prune all elements in a large probability distribution below the top K
#  Renormalize the distribution with only top K, and then sample n times out of that

def beamMultinomial(k, probs):
    
    tops = topk(k, probs)
    i = 0
    n = len(tops.keys())
    ary = np.zeros((n))
    idx = []
    for abs_idx,v in tops.iteritems():
        ary[i] = v
        idx.append(abs_idx)
        i += 1

    ary /= np.sum(ary)
    sample_idx = np.argmax(np.random.multinomial(1, ary))
    return idx[sample_idx]


