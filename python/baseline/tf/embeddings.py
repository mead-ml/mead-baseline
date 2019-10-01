from eight_mile.tf.embeddings import *
import tensorflow as tf
from baseline.tf.tfy import stacked_lstm

@register_embeddings(name='char-lstm')
class CharLSTMEmbeddings(TensorFlowEmbeddings):
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharLSTMEmbeddings, self).__init__(name=name, **kwargs)
        self._name = name
        self.scope = kwargs.get('scope', '{}/CharLUT'.format(self._name))
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self._weights = kwargs.get('weights')
        self.lstmsz = kwargs.get('lstmsz', 50)
        self.layers = kwargs.get('layers', 1)
        self.pdrop = kwargs.get('pdrop', 0.5)
        self.rnn_type = kwargs.get('rnn_type', 'blstm')
        self.x = None
        if self._weights is None:
            unif = kwargs.get('unif', 0.1)
            self._weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))

    def detached_ref(self):
        if self._weights is None:
            raise Exception('You must initialize `weights` in order to use this method.')
        return CharLSTMEmbeddings(
            name=self._name, vsz=self.vsz, dsz=self.dsz, scope=self.scope,
            finetune=self.finetune, lstmsz=self.lstmsz, layers=self.layers,
            dprop=self.pdrop, rnn_type=self.rnn_type, weights=self._weights,
        )

    def encode(self, x=None):
        if x is None:
            x = CharLSTMEmbeddings.create_placeholder(self._name)

        self.x = x
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            Wch = tf.get_variable(
                "Wch",
                initializer=tf.constant_initializer(self._weights, dtype=tf.float32, verify_shape=True),
                shape=[self.vsz, self.dsz],
                trainable=True
            )
            ech0 = tf.scatter_update(Wch, tf.constant(Offsets.PAD, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, self.dsz]))

            shape = tf.shape(x)
            B = shape[0]
            T = shape[1]
            W = shape[2]
            flat_chars = tf.reshape(x, [-1, W])
            word_lengths = tf.reduce_sum(tf.cast(tf.equal(flat_chars, Offsets.PAD), tf.int32), axis=1)
            with tf.control_dependencies([ech0]):
                embed_chars = tf.nn.embedding_lookup(Wch, flat_chars)

            fwd_lstm = stacked_lstm(self.lstmsz // 2, self.pdrop, self.layers)
            bwd_lstm = stacked_lstm(self.lstmsz // 2, self.pdrop, self.layers)
            _, rnn_state = tf.nn.bidirectional_dynamic_rnn(fwd_lstm, bwd_lstm, embed_chars, sequence_length=word_lengths, dtype=tf.float32)

            result = tf.concat([rnn_state[0][-1].h, rnn_state[1][-1].h], axis=1)
            return tf.reshape(result, [B, T, self.lstmsz])

    def get_dsz(self):
        return self.lstmsz

    def get_vsz(self):
        return self.vsz

    def get_config(self):
        config = super(CharLSTMEmbeddings, self).get_config()
        config['lstmsz'] = self.lstmsz
