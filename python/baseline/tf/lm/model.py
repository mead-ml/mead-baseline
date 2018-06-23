from baseline.tf.tfy import *
from baseline.model import create_lang_model
import json


class AbstractLanguageModel(object):

    def __init__(self):
        self.layers = None
        self.hsz = None
        self.rnntype = 'lstm'
        self.pkeep = None
        self.saver = None

    def save_using(self, saver):
        self.saver = saver

    def _rnnlm(self, inputs, vsz):

        rnnfwd = stacked_lstm(self.hsz, self.pkeep, self.layers)
        self.initial_state = rnnfwd.zero_state(self.batchsz, tf.float32)
        rnnout, state = tf.nn.dynamic_rnn(rnnfwd, inputs, initial_state=self.initial_state, dtype=tf.float32)
        self.final_state = state

        output = tf.reshape(tf.concat(rnnout, 1), [-1, self.hsz])

        softmax_w = tf.get_variable(
            "softmax_w", [self.hsz, vsz], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vsz], dtype=tf.float32)

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b, name="logits")

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def create_loss(self):
        with tf.variable_scope("Loss"):
            targets = tf.reshape(self.y, [-1])
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([tf.size(targets)], dtype=tf.float32)])
            loss = tf.reduce_sum(loss) / self.batchsz
            return loss

    def get_vocab(self, vocab_type='word'):
        pass


class WordLanguageModel(AbstractLanguageModel):

    def __init__(self):
        AbstractLanguageModel.__init__(self)

    def make_input(self, batch_dict, do_dropout=False):
        x = batch_dict['x']
        y = batch_dict['y']
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.x: x, self.y: y, self.pkeep: pkeep}
        return feed_dict

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        word_vec = embeddings['word']

        lm.batchsz = kwargs['batchsz']
        lm.mxlen = kwargs.get('mxlen', kwargs['nbptt'])
        lm.maxw = kwargs['maxw']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.x = kwargs.get('x', tf.placeholder(tf.int32, [None, lm.mxlen], name="x"))
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, lm.mxlen], name="y"))
        lm.rnntype = kwargs.get('rnntype', 'lstm')
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        pdrop = kwargs.get('pdrop', 0.5)
        lm.pdrop_value = pdrop
        lm.hsz = kwargs['hsz']
        lm.word_vocab = word_vec.vocab
        vsz = word_vec.vsz + 1

        with tf.name_scope("WordLUT"):
            Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")
            we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
            with tf.control_dependencies([we0]):
                wembed = tf.nn.embedding_lookup(Ww, lm.x, name="embeddings")

        inputs = tf.nn.dropout(wembed, lm.pkeep)
        ##inputs = tf.unstack(inputs, num=lm.mxlen, axis=1)
        lm.layers = kwargs.get('layers', kwargs.get('nlayers', 1))
        lm._rnnlm(inputs, vsz)
        return lm

    def get_vocab(self, vocab_type='word'):
        if vocab_type == 'word':
            return self.word_vocab
        return None

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"mxlen": self.mxlen, "maxw": self.maxw,
                 'hsz': self.hsz, 'batchsz': self.batchsz,
                 'layers': self.layers}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)


class CharCompLanguageModel(AbstractLanguageModel):

    def __init__(self):
        AbstractLanguageModel.__init__(self)

    def make_input(self, batch_dict, do_dropout=False):
        x = batch_dict['x']
        xch = batch_dict['xch']
        y = batch_dict['y']

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.x: x, self.xch: xch, self.y: y, self.pkeep: pkeep}
        return feed_dict

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        word_vec = embeddings['word']
        char_vec = embeddings['char']
        lm.batchsz = kwargs['batchsz']
        kwargs['mxlen'] = kwargs.get('mxlen', kwargs['nbptt'])
        lm.mxlen = kwargs['mxlen']
        lm.maxw = kwargs['maxw']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.x = kwargs.get('x', tf.placeholder(tf.int32, [None, lm.mxlen], name="x"))
        lm.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, lm.mxlen, lm.maxw], name="xch"))
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, lm.mxlen], name="y"))
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        lm.rnntype = kwargs.get('rnntype', 'lstm')
        vsz = word_vec.vsz + 1
        lm.char_vocab = char_vec.vocab
        lm.word_vocab = word_vec.vocab
        lm.pdrop_value = kwargs.get('pdrop', 0.5)
        lm.layers = kwargs.get('layers', kwargs.get('nlayers', 1))
        char_dsz = char_vec.dsz
        with tf.name_scope("CharLUT"):
            Wch = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch", trainable=True)
            ech0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))
            word_char, wchsz = pool_chars(lm.xch, Wch, ech0, char_dsz, **kwargs)

        lm.use_words = kwargs.get('use_words', False)
        if lm.use_words:
            with tf.name_scope("WordLUT"):
                Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")
                we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
                with tf.control_dependencies([we0]):
                    wembed = tf.nn.embedding_lookup(Ww, lm.x, name="embeddings")
                    word_char = tf.concat(values=[wembed, word_char], axis=2)

        inputs = tf.nn.dropout(word_char, lm.pkeep)
        inputs = tf.unstack(inputs, num=lm.mxlen, axis=1)
        lm.hsz = kwargs['hsz']
        lm._rnnlm(inputs, vsz)
        return lm

    def get_vocab(self, vocab_type='word'):
        if vocab_type == 'char':
            return self.char_vocab
        return None

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"mxlen": self.mxlen, "maxw": self.maxw, 'use_words': self.use_words,
                 'layers': self.layers}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(self.char_vocab, f)


BASELINE_LM_MODELS = {
    'default': WordLanguageModel.create,
    'convchar': CharCompLanguageModel.create
}

# TODO:
# BASELINE_LM_LOADERS = {
#    'default': WordLanguageModel.load,
#    'convchar': CharCompLanguageModel.load
# }


# TODO: move the scoping and weight initialization into the model itself
def create_model(embeddings, **kwargs):
    unif = kwargs['unif']

    if 'sess' not in kwargs:
        kwargs['sess'] = tf.Session()

    weight_initializer = tf.random_uniform_initializer(-unif, unif)
    with tf.variable_scope('Model', initializer=weight_initializer):
        lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm
