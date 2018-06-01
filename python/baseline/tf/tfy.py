import tensorflow as tf
import numpy as np
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial
import os


def _find_files_by_type(model_file, filetype):
    """Find all files by type, removing suffix

    we rely on the fact that vocab files end in .vocab.

    :return: the file names without the filetype.
    """
    matching_files = []

    filetype_ending = "." + filetype
    basepath = get_basepath_or_cwd(model_file)

    for filename in os.listdir(basepath):
        if filename.endswith(filetype_ending):
            filename_without_ending = filename[:-len(filetype_ending)]
            matching_files.append(os.path.join(basepath, filename_without_ending))

    if not matching_files:
        raise ValueError("no vocab files found in directory %s. \
Please specify the model as path-like. e.g. /data/model/model-name-1234" % basepath)

    return matching_files

def get_basepath_or_cwd(model_file):
    """
    inspects the model_file variable for a directory name.

    if no directory is found, returns current working dir.
    """
    basepath = os.path.dirname(model_file)
    if not os.path.isdir(basepath):
        basepath = os.getcwd()

    return basepath


def get_vocab_file_suffixes(model_file):

    """Because our operations assume knowledge of the model name, we
    only need to return the suffix appended onto the end of the model
    name in the file.

    we make the assumption that a suffix is denoted with a hyphen.

    e.g.  a vocab file name = tagger-model-tf-30803-word.vocab
          would return ['word']

    :param model_file: the nonspecific path to the model. this could be
                /data/model/<model_name>. we need to remove the model name.
    :return:
    """
    filenames = _find_files_by_type(model_file, 'vocab')
    # mmodel_file can be path like or a string for just the model name.
    name_parts = model_file.split('/')
    model_name = name_parts[-1]
    # the length of the name plus 1 for the hyphen separating the suffix.
    return [x.split('/')[-1][len(model_name)+1:] for x in filenames]


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

    # warm restarts in master, not in 1.5 yet
    #elif decay_type == 'sgdr':
    #    at_step = kwargs.get('bounds', 1000)
    #    decay_fn = lambda lr, global_step: tf.train.cosine_decay_restarts(lr, global_step, first_decay_steps=at_step)

    elif decay_type == 'cosine':
        at_step = kwargs.get('bounds', 1000)
        decay_fn = lambda lr, global_step: tf.train.cosine_decay(lr, global_step, at_step)

    elif decay_type == 'lincos':
        at_step = kwargs.get('bounds', 1000)
        decay_fn = lambda lr, global_step: tf.train.linear_cosine_decay(lr, global_step, at_step)

    elif decay_type == 'zaremba':
        boundaries = kwargs.get('bounds', None)
        decay_rate = float(kwargs.get('decay_rate', None))
        values = [eta/(decay_rate**i) for i in range(len(boundaries)+1)]
        print('Learning rate schedule:')
        print('B', len(boundaries), boundaries)
        print('V', len(values), values)
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


def dense_layer(output_layer_depth):
    output_layer = layers_core.Dense(output_layer_depth, use_bias=False, dtype=tf.float32, name="dense")
    return output_layer


def lstm_cell(hsz, forget_bias=1.0):
    return tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True)


def lstm_cell_w_dropout(hsz, pkeep, forget_bias=1.0):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(hsz, forget_bias=forget_bias, state_is_tuple=True), output_keep_prob=pkeep)


def stacked_lstm(hsz, pkeep, nlayers):
    return tf.contrib.rnn.MultiRNNCell([lstm_cell_w_dropout(hsz, pkeep) if i < nlayers - 1 else lstm_cell(hsz) for i in range(nlayers)], state_is_tuple=True)


def stacked_cnn(inputs, hsz, pkeep, nlayers, activation_fn=tf.nn.relu, filts=[5]):
    with tf.variable_scope("StackedCNN"):
        layers = []
        for filt in filts:
            layer = tf.nn.dropout(tf.layers.conv1d(inputs, hsz, filt, activation=activation_fn, padding="same", reuse=False), pkeep)

            for i in range(1, nlayers):
                layer = layer + tf.nn.dropout(tf.layers.conv1d(inputs, hsz, filt, activation=activation_fn, padding="same", reuse=False), pkeep)
            layers += [layer]

        return tf.concat(values=layers, axis=2)


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


def skip_conns(inputs, wsz_all, n, activation_fn=tf.nn.relu):
    for i in range(n):
        with tf.variable_scope("skip-%d" % i):
            W_p = tf.get_variable("W_p", [wsz_all, wsz_all])
            b_p = tf.get_variable("B_p", [1, wsz_all], initializer=tf.constant_initializer(0.0))
            proj = activation_fn(tf.matmul(inputs, W_p) + b_p, "skip_activation")

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


def parallel_conv(input_, filtsz, dsz, motsz, activation_fn=tf.nn.relu):
    """Do parallel convolutions with multiple filter widths and max-over-time pooling.

    :param input_: The inputs in the shape [B, T, H].
    :param filtsz: The list of filter widths to use.
    :param dsz: The depths of the input (H).
    :param motsz: The number of conv filters to use (can be an int or a list to allow for various sized filters)

    :Keyword Arguments:
    * *activation_fn* -- (``callable``) The activation function to apply after the convolution and bias add
    """
    if not isinstance(motsz, list):
        motsz = [motsz] * len(filtsz)
    DUMMY_AXIS = 1
    TIME_AXIS = 2
    FEATURE_AXIS = 3
    expanded = tf.expand_dims(input_, DUMMY_AXIS)
    mots = []
    for fsz, cmotsz in zip(filtsz, motsz):
        with tf.variable_scope('cmot-%s' % fsz):
            kernel_shape = [1, fsz, dsz, cmotsz]
            W = tf.get_variable('W', kernel_shape)
            b = tf.get_variable(
                'b', [cmotsz],
                initializer=tf.constant_initializer(0.0)
            )
            conv = tf.nn.conv2d(
                expanded, W,
                strides=[1, 1, 1, 1],
                padding="SAME", name="CONV"
            )
            activation = activation_fn(tf.nn.bias_add(conv, b), 'activation')
            mot = tf.reduce_max(activation, [TIME_AXIS], keep_dims=True)
            mots.append(mot)
    motsz_all = sum(motsz)
    combine = tf.reshape(tf.concat(values=mots, axis=FEATURE_AXIS), [-1, motsz_all])
    return combine


def char_word_conv_embeddings(char_vec, filtsz, char_dsz, wsz, activation_fn=tf.nn.tanh):
    combine = parallel_conv(char_vec, filtsz, char_dsz, wsz, activation_fn)
    wsz_all = wsz * len(filtsz)
    joined = skip_conns(combine, wsz_all, 1)
    return joined


def tf_activation(name):
    if name == "tanh":
            return tf.nn.tanh
    if name == "sigmoid":
        return tf.nn.sigmoid
    return tf.nn.relu


def char_word_conv_embeddings_var_fm(char_vec, filtsz, char_dsz, nfeat_factor, max_feat=200, activation_fn=tf.nn.tanh):
    nfeats = [min(nfeat_factor * fsz, max_feat) for fsz in filtsz]
    wsz_all = sum(nfeats)
    combine = parallel_conv(char_vec, filtsz, char_dsz, nfeats, activation_fn)
    joined = highway_conns(combine, wsz_all, 2)
    return joined


def pool_chars(xch, Wch, ce0, char_dsz, **kwargs):
    """Take in a tensor of characters (B x maxs x maxw) and do character convolution

    :param xch: TF tensor for input characters, (B x maxs x maxw)
    :param Wch: A character embeddings matrix
    :param ce0: A control dependency for the embeddings that keeps the <PAD> value 0
    :param char_dsz: The character embedding dsz
    :param kwargs:
    :return: The character compositional embedding and the number of hidden units as a tuple
    """
    wsz = kwargs.get('wsz', 30)
    filtsz = kwargs.get('cfiltsz', [3])
    mxlen = int(kwargs.get('maxs', kwargs.get('mxlen', 100)))
    mxwlen = kwargs.get('maxw', kwargs.get('mxwlen', 40))
    activation_type = kwargs.get('activation', 'tanh')
    with tf.variable_scope("Chars2Word"):
        with tf.control_dependencies([ce0]):
            char_bt_x_w = tf.reshape(xch, [-1, mxwlen])
            cembed = tf.nn.embedding_lookup(Wch, char_bt_x_w, name="embeddings")
            cmot = char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz,
                                             activation_fn=tf_activation(activation_type))
            word_char = tf.reshape(cmot, [-1, mxlen, len(filtsz) * wsz])

    return word_char, len(filtsz) * wsz


def shared_char_word(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        cembed = tf.nn.embedding_lookup(Wch, xch_i)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return char_word_conv_embeddings(cembed, filtsz, char_dsz, wsz)


def shared_char_word_var_fm(Wch, xch_i, filtsz, char_dsz, wsz, reuse):

    with tf.variable_scope("SharedCharWord", reuse=reuse):
        cembed = tf.nn.embedding_lookup(Wch, xch_i)
        if len(filtsz) == 0 or filtsz[0] == 0:
            return tf.reduce_sum(cembed, [1])
        return char_word_conv_embeddings_var_fm(cembed, filtsz, char_dsz, wsz)
