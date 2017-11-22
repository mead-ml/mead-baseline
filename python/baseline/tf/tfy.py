import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial


def optimizer(loss_fn, **kwargs):

    global_step = tf.Variable(0, trainable=False)
    clip = kwargs.get('clip', None)
    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_type = kwargs.get('decay_type', None)
    decay_fn = None

    if decay_type == 'piecewise':
        boundaries = kwargs.get('bounds', None)
        decay_values = kwargs.get('decay_values', None)
        decay_fn = lambda lr, global_step: tf.train.piecewise_constant(global_step, boundaries, decay_values)

    elif decay_type == 'staircase':
        at_step = int(kwargs.get('bounds', 16000))
        decay_rate = float(kwargs.get('decay_rate', 0.5))
        decay_fn = lambda lr, global_step: tf.train.exponential_decay(lr, global_step, at_step, decay_rate, staircase=True)

    elif decay_type == 'invtime':
        decay_rate = float(kwargs.get('decay_rate', 0.05))
        at_step = int(kwargs.get('bounds', 16000))
        decay_fn = lambda lr, global_step: tf.train.inverse_time_decay(lr, global_step, at_step, decay_rate, staircase=False)

    elif decay_type == 'zaremba':
        boundaries = kwargs.get('bounds', None)
        decay_rate = float(kwargs.get('decay_rate', None))
        values = [eta/(decay_rate**i) for i in range(len(boundaries))]
        print('Learning rate schedule:')
        print(boundaries)
        print(values)
        decay_fn = lambda lr, global_step: tf.train.piecewise_constant(global_step, boundaries, values)

    if optim == 'adadelta':
        print('adadelta', eta)
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)
    elif optim == 'adam':
        print('adam', eta)
        optz = lambda lr: tf.train.AdamOptimizer(lr)
    elif mom > 0:
        print('sgd-mom', eta, mom)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, mom)
    else:
        print('sgd')
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    print('clip', clip)
    print('decay', decay_fn)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz,
                                                        clip_gradients=clip, learning_rate_decay_fn=decay_fn)


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


def lstm_cell(hsz):
    return tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=0.0, state_is_tuple=True)


def lstm_cell_w_dropout(hsz, pkeep):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=1.0, state_is_tuple=True), output_keep_prob=pkeep)


def stacked_lstm(hsz, pkeep, nlayers):
    return tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pkeep) for _ in range(nlayers)], state_is_tuple=True)


def rnn_cell_w_dropout(hsz, pkeep, rnntype, st=None):
    if st is not None:
        cell = tf.contrib.rnn.BasicLSTMCell(hsz, state_is_tuple=st) if rnntype.endswith('lstm') else tf.contrib.rnn.GRUCell(hsz)
    else:
        cell = tf.contrib.rnn.LSTMCell(hsz) if rnntype.endswith('lstm') else tf.contrib.rnn.GRUCell(hsz)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=pkeep)


def multi_rnn_cell_w_dropout(hsz, pkeep, rnntype, num_layers):
    return tf.contrib.rnn.MultiRNNCell([rnn_cell_w_dropout(hsz, pkeep, rnntype) for _ in range(num_layers)], state_is_tuple=True)


# This function should never be used for decoding.  It exists only so that the training model can greedily decode
# It is super slow and doesnt use maintain a beam of hypotheses
def show_examples_tf(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    batch_dict = es[si]
    src_array = batch_dict['src']
    tgt_array = batch_dict['dst']
    src_len = batch_dict['src_len']

    if max_examples > 0:
        max_examples = min(max_examples, src_array.shape[0])
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']

    for src_len_i, src_i, tgt_i in zip(src_len, src_array, tgt_array):

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
            output = model.step({'src': src_i, 'src_len': src_len_i, 'dst': dst_i, 'dst_len': tgt_len_i})[j]
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


def skip_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu")

        inputs = inputs + proj
    return inputs


def highway_conns(inputs, wsz_all, n):
    for i in range(n):
        with tf.variable_scope("highway-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = tf.nn.relu(tf.matmul(inputs, W_p) + b_p, "relu-proj")

            W_t = tf.get_variable("W_t", [wsz_all, wsz_all])
            b_t = tf.get_variable("B_t", [1, wsz_all], initializer=tf.constant_initializer(-2.0))
            transform = tf.nn.sigmoid(tf.matmul(inputs, W_t) + b_t, "sigmoid-transform")

        inputs = tf.multiply(transform, proj) + tf.multiply(inputs, 1 - transform)
    return inputs


def char_word_conv_embeddings(char_vec, filtsz, char_dsz, wsz):

    expanded = tf.expand_dims(char_vec, -1)
    mots = []
    for i, fsz in enumerate(filtsz):
        with tf.variable_scope('cmot-%s' % fsz):

            kernel_shape = [fsz, char_dsz, 1, wsz]

            # Weight tying
            W = tf.get_variable("W", kernel_shape)
            b = tf.get_variable("b", [wsz], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(expanded,
                                W, strides=[1,1,1,1],
                                padding="VALID", name="conv")

            activation = tf.nn.relu(tf.nn.bias_add(conv, b), "activation")

            mot = tf.reduce_max(activation, [1], keep_dims=True)
            # Add back in the dropout
            mots.append(mot)

    wsz_all = wsz * len(mots)
    combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])

    # joined = highway_conns(combine, wsz_all, 1)
    joined = skip_conns(combine, wsz_all, 1)
    return joined



def char_word_conv_embeddings_var_fm(char_vec, filtsz, char_dsz, nfeat_factor, max_feat=200):

    expanded = tf.expand_dims(char_vec, -1)
    mots = []
    wsz_all = 0
    # wsz is feature factor
    for i, fsz in enumerate(filtsz):

        nfeat = min(nfeat_factor * fsz, max_feat)
        wsz_all += nfeat
        with tf.variable_scope('cmot-%s' % fsz):

            kernel_shape = [fsz, char_dsz, 1, nfeat]

            # Weight tying
            W = tf.get_variable("W", kernel_shape)
            b = tf.get_variable("b", [nfeat], initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(expanded,
                                W, strides=[1, 1, 1, 1],
                                padding="VALID", name="conv")

            activation = tf.nn.tanh(tf.nn.bias_add(conv, b), "activation")

            mot = tf.reduce_max(activation, [1], keep_dims=True)
            # Add back in the dropout
            mots.append(mot)

    combine = tf.reshape(tf.concat(values=mots, axis=3), [-1, wsz_all])
    joined = highway_conns(combine, wsz_all, 2)
    return joined


def shared_char_word(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        # Zeropad the letters out to half the max filter size, to account for
        # wide convolution.  This way we don't have to explicitly pad the
        # data upfront, which means our Y sequences can be assumed not to
        # start with zeros
        mxfiltsz = np.max(filtsz)
        halffiltsz = mxfiltsz // 2
        zeropad = tf.pad(xch_i, [[0, 0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz)


def shared_char_word_var_fm(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        # Zeropad the letters out to half the max filter size, to account for
        # wide convolution.  This way we don't have to explicitly pad the
        # data upfront, which means our Y sequences can be assumed not to
        # start with zeros
        mxfiltsz = np.max(filtsz)
        halffiltsz = mxfiltsz // 2
        zeropad = tf.pad(xch_i, [[0, 0], [halffiltsz, halffiltsz]], "CONSTANT")
        cembed = tf.nn.embedding_lookup(Wch, zeropad)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return char_word_conv_embeddings_var_fm(cembed, filtsz, char_dsz, wsz)
