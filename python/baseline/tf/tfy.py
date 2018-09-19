import os
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from baseline.utils import lookup_sentence, beam_multinomial, crf_mask as crf_m


def _add_ema(model, decay):
    """Create ops needed to track EMA when training.

    :param model: The model with a `.sess` we want to track.
    :param decay: float, Decay to use in the EMA

    :returns:
        ema_op: The update op. This applies the ema to each variable. Should be
           set as a control dependency on the training op.
        load: Op to copy emas to the variables.
        restore_var: Op to copy the original variables back from the EMA ones.

    Note:
        If you run the load op multiple times then the backup variables will be
        replaced by the ema variables.

        Currently there was a bug I haven't been able to fix. I haven't found why
        but sometimes when you run it with a tf.cond you get this error.
        `tensorflow.python.framework.errors_impl.InvalidArgumentError: Retval[0] does not have value`
        The stop gap is to remove this which means if you run load multiple times
        it will over write the backup variables with ema values.

        The load op is set up to automatically save the normal parameters when
        you load the ema's in.
    """
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    model_vars = model.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.variable_scope("BackupVariables"):
        backup_vars = [
            tf.get_variable(
                var.op.name,
                dtype=var.value().dtype,
                trainable=False,
                initializer=var.initialized_value()
            ) for var in model_vars
        ]
    ema_op = ema.apply(model_vars)

    save_back_up = tf.group(*(
        tf.assign(back, var.read_value())
        for var, back in zip(model_vars, backup_vars)
    ), name='save_backups')

    with tf.control_dependencies([save_back_up]):
        load = tf.group(*(
            tf.assign(var, ema.average(var).read_value())
            for var in model_vars
        ), name="load_ema")

    restore_vars = tf.group(*(
        tf.assign(var, back.read_value())
        for var, back in zip(model_vars, backup_vars)
    ), name="restore_backups")

    return ema_op, load, restore_vars


def crf_mask(vocab, span_type, s_idx, e_idx, pad_idx=None):
    """Create a CRF Mask.

    Returns a mask with invalid moves as 0 and valid moves as 1.
    """
    return tf.constant(crf_m(vocab, span_type, s_idx, e_idx, pad_idx).T)

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
    model_name = model_file.split('/')[-1]
    basepath = get_basepath_or_cwd(model_file)
    full_base = os.path.join(basepath, model_name)
    # the length of the name plus 1 for the hyphen separating the suffix.
    return [x[len(full_base)+1:] for x in filenames if x.startswith(full_base)]


def optimizer(loss_fn, **kwargs):

    global_step = tf.Variable(0, trainable=False)
    clip = kwargs.get('clip', None)
    mom = kwargs.get('mom', 0.9)
    optim = kwargs.get('optim', 'sgd')
    eta = kwargs.get('eta', kwargs.get('lr', 0.01))
    decay_type = kwargs.get('decay_type', None)
    decay_fn = None
    colocate_gradients_with_ops = bool(kwargs.get('colocate_gradients_with_ops', False))
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
    elif optim == 'rmsprop':
        print('rmsprop', eta)
        optz = lambda lr: tf.train.RMSPropOptimizer(lr, momentum=mom)
    elif mom > 0:
        print('sgd-mom', eta, mom)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, mom)
    else:
        print('sgd')
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    print('clip', clip)
    print('decay', decay_fn)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz,
                                                        colocate_gradients_with_ops=colocate_gradients_with_ops,
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


def create_show_examples_tf(src_key):
    # This function should never be used for decoding.  It exists only so that the training model can greedily decode
    def show_examples_tf(model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
        si = np.random.randint(0, len(es))

        batch_dict = es[si]
        GO = embed2.vocab['<GO>']
        EOS = embed2.vocab['<EOS>']
        i = 0
        src_lengths_key = '{}_lengths'.format(src_key)

        while True:

            example = {}
            for k in batch_dict.keys():
                if i >= len(batch_dict[k]):
                    return
                example[k] = batch_dict[k][i]
            print('========================================================================')

            src_i = example[src_key]
            src_len_i = example[src_lengths_key]
            dst_i = example['dst']

            sent = lookup_sentence(rlut1, src_i, reverse=reverse)
            print('[OP] %s' % sent)
            sent = lookup_sentence(rlut2, dst_i)
            print('[Actual] %s' % sent)
            dst_i = np.zeros((1, mxlen))
            example['dst'] = dst_i
            src_i = src_i[np.newaxis, :]
            example[src_key] = src_i
            example[src_lengths_key] = np.array([src_len_i])
            next_value = GO
            for j in range(mxlen):
                dst_i[0, j] = next_value
                tgt_len_i = np.array([j+1])
                example['dst_lengths'] = tgt_len_i
                output = model.step(example)[j]
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
            i += 1
            if i == max_examples:
                return
    return show_examples_tf

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
    :param activation_fn: The activation function to use (`default=tf.nn.relu`)
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
            mot = tf.reduce_max(activation, [TIME_AXIS], keepdims=True)
            mots.append(mot)
    motsz_all = sum(motsz)
    combine = tf.reshape(tf.concat(values=mots, axis=FEATURE_AXIS), [-1, motsz_all])
    return combine, motsz_all


def tf_activation(name):
    if name == "tanh":
            return tf.nn.tanh
    if name == "sigmoid":
        return tf.nn.sigmoid
    return tf.nn.relu


def char_word_conv_embeddings(char_vec, filtsz, char_dsz, nfeats, activation_fn=tf.nn.tanh, gating=skip_conns, num_gates=1):
    combine, wsz_all = parallel_conv(char_vec, filtsz, char_dsz, nfeats, activation_fn)
    joined = gating(combine, wsz_all, num_gates)
    return joined, wsz_all


def pool_chars(x_char, Wch, ce0, char_dsz, **kwargs):
    """Take in a tensor of characters (B x maxs x maxw) and do character convolution

    :param x_char: TF tensor for input characters, (B x maxs x maxw)
    :param Wch: A character embeddings matrix
    :param ce0: A control dependency for the embeddings that keeps the <PAD> value 0
    :param char_dsz: The character embedding dsz
    :param kwargs:
    :return: The character compositional embedding and the number of hidden units as a tuple
    """
    filtsz = kwargs.get('cfiltsz', [3])
    if 'nfeat_factor' in kwargs:
        max_feat = kwargs.get('max_feat', 200)
        nfeats = [min(kwargs['nfeat_factor'] * fsz, max_feat) for fsz in filtsz]
    else:
        nfeats = kwargs.get('wsz', 30)
    mxlen = tf.shape(x_char)[1]
    gating = kwargs.get('gating', "skip")
    gating_fn = highway_conns if gating.startswith('highway') else skip_conns
    num_gates = int(kwargs.get('num_gates', 1))
    # print(gating_fn, num_gates)
    activation_type = kwargs.get('activation', 'tanh')
    with tf.variable_scope("Chars2Word"):
        with tf.control_dependencies([ce0]):
            mxwlen = tf.shape(x_char)[-1]
            char_bt_x_w = tf.reshape(x_char, [-1, mxwlen])
            cembed = tf.nn.embedding_lookup(Wch, char_bt_x_w, name="embeddings")
            cmot, num_filts = char_word_conv_embeddings(cembed, filtsz, char_dsz, nfeats,
                                                        activation_fn=tf_activation(activation_type),
                                                        gating=gating_fn,
                                                        num_gates=num_gates)
            word_char = tf.reshape(cmot, [-1, mxlen, num_filts])

    return word_char, num_filts


def embed(x, vsz, dsz, initializer, finetune=True, scope="LUT"):
    with tf.variable_scope(scope):
        W = tf.get_variable("W",
                            initializer=initializer,
                            shape=[vsz, dsz], trainable=finetune)
        e0 = tf.scatter_update(W, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, dsz]))
        with tf.control_dependencies([e0]):
            word_embeddings = tf.nn.embedding_lookup(W, x)

    return word_embeddings
