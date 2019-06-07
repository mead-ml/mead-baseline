import os
import tensorflow as tf
import tensorflow_hub as hub
from baseline.utils import write_json, read_json, listify
from baseline.embeddings import register_embeddings
from baseline.tf.embeddings import TensorFlowEmbeddings
from baseline.vectorizers import AbstractVectorizer, register_vectorizer, _token_iterator
import numpy as np
import tensorflow as tf
import h5py
import json
import re
import glob
import random
import collections
from typing import List

"""
ELMo embeddings (hub-based and local weights both supported)

Large portions of this file are copied from https://github.com/allenai/bilm-tf
Several pieces were modified where it makes things more convenient/efficient
"""

DTYPE = 'float32'
DTYPE_INT = 'int64'
START_TOKEN = '<GO>'  # <S>
END_TOKEN = '<EOS>'  # </S>
UNK_TOKEN = '<UNK>'  # <UNK>
SPECIAL_CHARS = set([START_TOKEN, END_TOKEN, UNK_TOKEN])
ELMO_MXWLEN = 50

class Vocabulary(object):
    """
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    """
    def __init__(self, known_vocab):
        """
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens defined as ``START_TOKEN``, ``END_TOKEN`` and ``UNK_TOKEN`` above.
        """
        self._id_to_word = []
        self._word_to_id = {}

        self._bos = 0
        self._eos = 1
        self._unk = 2
        self._id_to_word.append(START_TOKEN)
        self._id_to_word.append(END_TOKEN)
        self._id_to_word.append(UNK_TOKEN)
        self._word_to_id[START_TOKEN] = self._bos
        self._word_to_id[END_TOKEN] = self._eos
        self._word_to_id[UNK_TOKEN] = self._unk
        # VALUES = ["<PAD>", "<GO>", "<EOS>", "<UNK>"]

        idx = self._unk + 1


        # if they already exist in this vocab, remove to ensure that they are always placed at the low offsets
        #known_vocab.pop(START_TOKEN, None)
        #known_vocab.pop(END_TOKEN, None)
        #known_vocab.pop(UNK_TOKEN, None)

        for word_name, count in known_vocab.items():

            if word_name == '!!!MAXTERMID' or word_name in SPECIAL_CHARS:
                continue

            self._id_to_word.append(word_name)
            self._word_to_id[word_name] = idx
            idx += 1

        # check to ensure file has special tokens
        #if validate_file:
        #    if self._bos == -1 or self._eos == -1 or self._unk == -1:
        #        raise ValueError("Ensure the vocabulary file has "
        #                         "{} {} {} tokens".format(START_TOKEN, END_TOKEN, UNK_TOKEN))

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, known_vocab, max_word_length=ELMO_MXWLEN, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(known_vocab, **kwargs)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length],
                                       dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32)
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        """
        Encode the sentence as a white space delimited string of tokens.
        """
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                         for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])


class BidirectionalLanguageModel(object):
    def __init__(
            self,
            options_file,
            weight_file,
            use_character_inputs=True,
            embedding_weight_file=None,
            max_batch_size=128,
    ):
        """
        Creates the language model computational graph and loads weights

        Two options for input type:
            (1) To use character inputs (paired with Batcher)
                pass use_character_inputs=True, and ids_placeholder
                of shape (None, None, max_characters_per_token)
                to __call__
            (2) To use token ids as input (paired with TokenBatcher),
                pass use_character_inputs=False and ids_placeholder
                of shape (None, None) to __call__.
                In this case, embedding_weight_file is also required input

        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        use_character_inputs: if True, then use character ids as input,
            otherwise use token ids
        max_batch_size: the maximum allowable batch size
        """
        if isinstance(options_file, dict):
            options = options_file
        else:
            with open(options_file, 'r') as fin:
                options = json.load(fin)

        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )

        self._options = options
        self._weight_file = weight_file
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self._ops = {}
        self._graphs = {}

    def __call__(self, ids_placeholder):
        """
        Given the input character ids (or token ids), returns a dictionary
            with tensorflow ops:

            {'lm_embeddings': embedding_op,
             'lengths': sequence_lengths_op,
             'mask': op to compute mask}

        embedding_op computes the LM embeddings and is shape
            (None, 3, None, 1024)
        lengths_op computes the sequence lengths and is shape (None, )
        mask computes the sequence mask and is shape (None, None)

        ids_placeholder: a tf.placeholder of type int32.
            If use_character_inputs=True, it is shape
                (None, None, max_characters_per_token) and holds the input
                character ids for a batch
            If use_character_input=False, it is shape (None, None) and
                holds the input token ids for a batch
        """
        if ids_placeholder in self._ops:
            # have already created ops for this placeholder, just return them
            ret = self._ops[ids_placeholder]

        else:
            # need to create the graph
            if len(self._ops) == 0:
                # first time creating the graph, don't reuse variables
                lm_graph = BidirectionalLanguageModelGraph(
                    self._options,
                    self._weight_file,
                    ids_placeholder,
                    embedding_weight_file=self._embedding_weight_file,
                    use_character_inputs=self._use_character_inputs,
                    max_batch_size=self._max_batch_size)
            else:
                with tf.variable_scope('', reuse=True):
                    lm_graph = BidirectionalLanguageModelGraph(
                        self._options,
                        self._weight_file,
                        ids_placeholder,
                        embedding_weight_file=self._embedding_weight_file,
                        use_character_inputs=self._use_character_inputs,
                        max_batch_size=self._max_batch_size)

            ops = self._build_ops(lm_graph)
            self._ops[ids_placeholder] = ops
            self._graphs[ids_placeholder] = lm_graph
            ret = ops

        return ret

    def _build_ops(self, lm_graph):
        with tf.control_dependencies([lm_graph.update_state_op]):
            # get the LM embeddings
            token_embeddings = lm_graph.embedding
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]

            n_lm_layers = len(lm_graph.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [lm_graph.lstm_outputs['forward'][i],
                         lm_graph.lstm_outputs['backward'][i]],
                        axis=-1
                    )
                )

            # The layers include the BOS/EOS tokens.  Remove them
            sequence_length_wo_bos_eos = lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []
            for layer in layers:
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    lm_graph.sequence_lengths - 1,
                    seq_axis=1,
                    batch_axis=0,
                    )
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

            # get the mask op without bos/eos.
            # tf doesn't support reversing boolean tensors, so cast
            # to int then back
            mask_wo_bos_eos = tf.cast(lm_graph.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                lm_graph.sequence_lengths - 1,
                seq_axis=1,
                batch_axis=0,
                )
            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                sequence_length_wo_bos_eos,
                seq_axis=1,
                batch_axis=0,
            )
            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

        return {
            'lm_embeddings': lm_embeddings,
            'lengths': sequence_length_wo_bos_eos,
            'token_embeddings': lm_graph.embedding,
            'mask': mask_wo_bos_eos,
        }


def _pretrained_initializer(varname, weight_file, embedding_weight_file=None):
    """
    We'll stub out all the initializers in the pretrained LM with
    a function that loads the weights from the file
    """
    weight_name_map = {}
    for i in range(2):
        for j in range(8):  # if we decide to add more layers
            root = 'RNN_{}/RNN/MultiRNNCell/Cell{}'.format(i, j)
            weight_name_map[root + '/rnn/lstm_cell/kernel'] = \
                root + '/LSTMCell/W_0'
            weight_name_map[root + '/rnn/lstm_cell/bias'] = \
                root + '/LSTMCell/B'
            weight_name_map[root + '/rnn/lstm_cell/projection/kernel'] = \
                root + '/LSTMCell/W_P_0'

    # convert the graph name to that in the checkpoint
    varname_in_file = varname[5:]
    if varname_in_file.startswith('RNN'):
        varname_in_file = weight_name_map[varname_in_file]

    if varname_in_file == 'embedding':
        with h5py.File(embedding_weight_file, 'r') as fin:
            # Have added a special 0 index for padding not present
            # in the original model.
            embed_weights = fin[varname_in_file][...]
            weights = np.zeros(
                (embed_weights.shape[0] + 1, embed_weights.shape[1]),
                dtype=DTYPE
            )
            weights[1:, :] = embed_weights
    else:
        with h5py.File(weight_file, 'r') as fin:
            if varname_in_file == 'char_embed':
                # Have added a special 0 index for padding not present
                # in the original model.
                char_embed_weights = fin[varname_in_file][...]
                weights = np.zeros(
                    (char_embed_weights.shape[0] + 1,
                     char_embed_weights.shape[1]),
                    dtype=DTYPE
                )
                weights[1:, :] = char_embed_weights
            else:
                weights = fin[varname_in_file][...]

    # Tensorflow initializers are callables that accept a shape parameter
    # and some optional kwargs
    def ret(shape, **kwargs):
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret


class BidirectionalLanguageModelGraph(object):
    """
    Creates the computational graph and holds the ops necessary for runnint
    a bidirectional language model
    """
    def __init__(self, options, weight_file, ids_placeholder,
                 use_character_inputs=True, embedding_weight_file=None,
                 max_batch_size=128):

        self.options = options
        self._max_batch_size = max_batch_size
        self.ids_placeholder = ids_placeholder
        self.use_character_inputs = use_character_inputs

        # this custom_getter will make all variables not trainable and
        # override the default initializer
        def custom_getter(getter, name, *args, **kwargs):
            kwargs['trainable'] = False
            kwargs['initializer'] = _pretrained_initializer(
                name, weight_file, embedding_weight_file
            )
            return getter(name, *args, **kwargs)

        if embedding_weight_file is not None:
            # get the vocab size
            with h5py.File(embedding_weight_file, 'r') as fin:
                # +1 for padding
                self._n_tokens_vocab = fin['embedding'].shape[0] + 1
        else:
            self._n_tokens_vocab = None

        with tf.variable_scope('bilm', custom_getter=custom_getter):
            self._build()

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()
        self._build_lstms()

    def _build_word_char_embeddings(self):
        """
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        """
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 262:
            raise Exception(
                "Set n_characters=262 after training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "char_embed", [n_chars, char_embed_dim],
                dtype=DTYPE,
                initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                         self.ids_placeholder)

        # the convolutions
        def make_convolutions(inp):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                        inp, w,
                        strides=[1, 1, 1, 1],
                        padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                        conv, [1, 1, max_chars-width+1, 1],
                        [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        embedding = make_convolutions(self.char_embedding)

        # for highway and projection layers
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            embedding = tf.reshape(embedding, [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                W_proj_cnn = tf.get_variable(
                    "W_proj", [n_filters, projection_dim],
                    initializer=tf.random_normal_initializer(
                        mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                    dtype=DTYPE)
                b_proj_cnn = tf.get_variable(
                    "b_proj", [projection_dim],
                    initializer=tf.constant_initializer(0.0),
                    dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)

        # finally project down if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        # at last assign attributes for remainder of the model
        self.embedding = embedding


    def _build_word_embeddings(self):
        projection_dim = self.options['lstm']['projection_dim']

        # the word embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "embedding", [self._n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.ids_placeholder)


    def _build_lstms(self):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)

        # parse the options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm']['use_skip_connections']

        # the sequence lengths from input mask
        if self.use_character_inputs:
            mask = tf.reduce_any(self.ids_placeholder > 0, axis=2)
        else:
            mask = self.ids_placeholder > 0
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        batch_size = tf.shape(sequence_lengths)[0]

        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}

        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    self.embedding,
                    sequence_lengths,
                    seq_axis=1,
                    batch_axis=0
                )

            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = lstm_cell.state_size
                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(
                        tf.zeros([self._max_batch_size, dim]),
                        trainable=False
                    )
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(
                    i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        layer_input,
                        sequence_length=sequence_lengths,
                        initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            *batch_init_states),
                    )

                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)
                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            layer_output,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )

                with tf.control_dependencies([layer_output]):
                    # update the initial states
                    for i in range(2):
                        new_state = tf.concat(
                            [final_state[i][:batch_size, :],
                             init_states[i][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)

                layer_input = layer_output

        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)


@register_vectorizer(name='elmo')
class ELMoVectorizer(AbstractVectorizer):
    def __init__(self, transform_fn=None, **kwargs):
        super(ELMoVectorizer, self).__init__(transform_fn=transform_fn)
        self.mxlen = kwargs.get('mxlen', 128)
        self.mxwlen = ELMO_MXWLEN
        self.max_seen = 0
        self.vocab = None

    def get_dims(self):
        return self.mxlen, self.mxwlen

    def count(self, tokens):
        seen = 0
        self.max_seen_char = 0
        counter = collections.Counter()
        for tok in self.iterable(tokens):
            counter[tok] += 1
            self.max_seen_char = max(len(tok), self.max_seen_char)
            seen += 1
        self.max_seen = max(self.max_seen, seen)
        return counter

    def run(self, tokens, vocab):

        if self.mxlen < 0:
            self.mxlen = self.max_seen

        if self.vocab is None:
            self.vocab = UnicodeCharsVocabulary(vocab)
        token_list = [t for t in self.iterable(tokens)]
        length = min(len(token_list), self.mxlen) + 2
        vec = np.zeros((self.mxlen + 2, self.mxwlen), dtype=np.int32)
        char_ids_without_mask = self.vocab.encode_chars(token_list, split=False)
        # add one so that 0 is the mask value
        vec[:length, :] = char_ids_without_mask[:length, :] + 1

        return vec, length


@register_vectorizer(name='dict_elmo')
class DictELMoVectorizer(ELMoVectorizer):
    def __init__(self, **kwargs):
        super(DictELMoVectorizer, self).__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        return _token_iterator(self, tokens)


def weight_layers(name, bilm_ops, l2_coef=None,
                  use_top_only=False, do_layer_norm=False):
    """
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES

    Input:
        name = a string prefix used for the trainable variable names
        bilm_ops = the tensorflow ops returned to compute internal
            representations from a biLM.  This is the return value
            from BidirectionalLanguageModel(...)(ids_placeholder)
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.
        use_top_only: if True, then only use the top layer.
        do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing

    Output:
        {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
        }
    """
    def _l2_regularizer(weights):
        if l2_coef is not None:
            return l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    # Get ops for computing LM embeddings and mask
    # This was modified to stop gradient flow
    lm_embeddings = tf.stop_gradient(bilm_ops['lm_embeddings'])
    mask = bilm_ops['mask']

    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2
                                     ) / N
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )

        if use_top_only:
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
            # just the top layer
            sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
            # no regularization
            reg = 0.0
        else:
            W = tf.get_variable(
                '{}_ELMo_W'.format(name),
                shape=(n_lm_layers, ),
                initializer=tf.zeros_initializer,
                regularizer=_l2_regularizer,
                trainable=True,
            )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
            )
            # split LM layers
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

            # compute the weighted, normalized LM activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if do_layer_norm:
                    pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                else:
                    pieces.append(w * tf.squeeze(t, squeeze_dims=1))
            sum_pieces = tf.add_n(pieces)

            # get the regularizer
            reg = [
                r for r in tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_ELMo_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by gamma
        gamma = tf.get_variable(
            '{}_ELMo_gamma'.format(name),
            shape=(1, ),
            initializer=tf.ones_initializer,
            regularizer=None,
            trainable=True,
        )
        weighted_lm_layers = sum_pieces * gamma

        ret = {'weighted_op': weighted_lm_layers, 'regularization_op': reg}

    return ret


@register_embeddings(name='elmo-embed')
class ELMoEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder('int32', shape=(None, None, ELMO_MXWLEN), name=name)

    def __init__(self, name, embed_file=None, known_vocab=None, **kwargs):
        super(ELMoEmbeddings, self).__init__(name=name, **kwargs)

        # options file
        self.weight_file = embed_file
        elmo_config = embed_file.replace('weights.hdf5', 'options.json')
        elmo_config = read_json(elmo_config)
        self.dsz = kwargs.get('dsz', 2*int(elmo_config['lstm']['projection_dim']))
        self.model = BidirectionalLanguageModel(elmo_config, self.weight_file)
        self.known_vocab = known_vocab
        self.vocab = UnicodeCharsVocabulary(known_vocab)

        assert self.dsz == 2*int(elmo_config['lstm']['projection_dim'])

    @property
    def vsz(self):
        return self.vocab.size

    def get_vocab(self):
        return self.vocab._word_to_id

    def get_dsz(self):
        return self.dsz

    @classmethod
    def load(cls, _, **kwargs):
        c = cls("elmo", **kwargs)
        return c

    def encode(self, x=None):
        if x is None:
            x = ELMoEmbeddings.create_placeholder(self.name)
        self.x = x

        context_embeddings_op = self.model(self.x)
        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
        elmo_weighting = elmo_context_input['weighted_op']
        return elmo_weighting

    def get_config(self):
        config = super(ELMoEmbeddings, self).get_config()
        config['embed_file'] = self.weight_file
        config['known_vocab'] = self.known_vocab
        return config


@register_embeddings(name='elmo')
class ELMoHubEmbeddings(TensorFlowEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.string, [None, None], name=name)

    def __init__(self, name, **kwargs):
        super(ELMoHubEmbeddings, self).__init__(**kwargs)
        self.vsz = None
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.cache_dir = kwargs.get('cache_dir')
        if self.cache_dir is not None:
            os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser(self.cache_dir)
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self.finetune)

    def encode(self, x=None):
        if x is None:
            x = ELMoHubEmbeddings.create_placeholder(self.name)
        self.x = x
        empty_count = tf.reduce_sum(tf.cast(tf.equal(self.x, ''), tf.int32), axis=1)
        T = tf.shape(self.x)[1]
        lengths = T - empty_count
        return self.elmo(
            inputs={
                'tokens': self.x,
                'sequence_len': lengths,
            },
            signature="tokens", as_dict=True)['elmo']

    def detached_ref(self):
        return ELMoHubEmbeddings(
            self.name, dsz=self.dsz, vsz=self.vsz, finetune=self.finetune, cache_dir=self.cache_dir
        )


@register_embeddings(name='elmo-pooled')
class ELMoPooledEmbeddings(ELMoEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.string, [None, None], name=name)

    def _create_conv_pooling(self, dsz, cmotsz, filtsz):
        from baseline.tf.tfy import parallel_conv

        def conv_pooling(embeddings, **kwargs):
            combine, _ = parallel_conv(embeddings, filtsz, dsz, cmotsz)
            return combine
        return conv_pooling

    def __init__(self, name, **kwargs):
        super(ELMoPooledEmbeddings, self).__init__(name, **kwargs)
        operator = kwargs.get('pooling', 'mean')
        if operator == 'max':
            self.pool_op = tf.reduce_max
        elif operator == 'sum':
            self.pool_op = tf.reduce_sum
        elif operator == 'conv':
            self.pool_op = self._create_conv_pooling(self.dsz, kwargs.get('cmotsz', 100), kwargs.get('filtsz', [3, 4, 5]))

        else:
            self.pool_op = tf.reduce_mean

    def encode(self, x=None):
        embeddings = super(ELMoPooledEmbeddings, self).encode(x)
        pooled = self.pool_op(embeddings, axis=1, keepdims=False, name='elmo_pooling')
        return pooled
