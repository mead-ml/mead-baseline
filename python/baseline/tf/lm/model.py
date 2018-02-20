from baseline.tf.tfy import *
from baseline.model import create_lang_model
import json


class AbstractLanguageModel(object):

    def __init__(self):
        pass

    def save_using(self, saver):
        self.saver = saver

    def _rnnlm(self, hsz, nlayers, inputs, vsz):

        def cell():
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(hsz), output_keep_prob=self.pkeep)

        cell = tf.contrib.rnn.MultiRNNCell(
            [cell() for _ in range(nlayers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batchsz, tf.float32)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self.initial_state, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hsz])

        softmax_w = tf.get_variable(
            "softmax_w", [hsz, vsz], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vsz], dtype=tf.float32)

        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b, name="logits")
        self.final_state = state

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

    @classmethod
    def create(cls, batchsz, nbptt, maxw, **kwargs):
        pass

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
        lm.nbptt = kwargs['nbptt']
        lm.maxw = kwargs['maxw']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.x = kwargs.get('x', tf.placeholder(tf.int32, [None, lm.nbptt], name="x"))
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, lm.nbptt], name="y"))
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        pdrop = kwargs.get('pdrop', 0.5)
        lm.pdrop_value = pdrop


        hsz = kwargs['hsz']
        lm.word_vocab = word_vec.vocab
        vsz = word_vec.vsz + 1

        with tf.name_scope("WordLUT"):
            Ww = tf.Variable(tf.constant(word_vec.weights, dtype=tf.float32), name="W")
            we0 = tf.scatter_update(Ww, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, word_vec.dsz]))
            with tf.control_dependencies([we0]):
                wembed = tf.nn.embedding_lookup(Ww, lm.x, name="embeddings")

        inputs = tf.nn.dropout(wembed, lm.pkeep)
        inputs = tf.unstack(inputs, num=lm.nbptt, axis=1)
        nlayers = kwargs.get('layer', kwargs.get('nlayers', 1))
        lm._rnnlm(hsz, nlayers, inputs, vsz)
        return lm

    def get_vocab(self, vocab_type='word'):
        if vocab_type == 'word':
            return self.word_vocab
        return None

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)
        with open(basename + '-batch_dims.json', 'w') as f:
            json.dump({'batchsz': self.batchsz, 'nbptt': self.nbptt, 'maxw': self.maxw}, f)


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
        lm.nbptt = kwargs['nbptt']
        lm.maxw = kwargs['maxw']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.x = kwargs.get('x', tf.placeholder(tf.int32, [None, lm.nbptt], name="x"))
        lm.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, lm.nbptt, lm.maxw], name="xch"))
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, lm.nbptt], name="y"))
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))

        filtsz = kwargs['cfiltsz']

        vsz = word_vec.vsz + 1
        lm.char_vocab = char_vec.vocab
        lm.pdrop_value = kwargs.get('pdrop', 0.5)
        char_dsz = char_vec.dsz
        Wc = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch")
        ce0 = tf.scatter_update(Wc, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

        wsz = kwargs['wsz']
        nlayers = kwargs.get('layer', kwargs.get('nlayers', 1))
        with tf.control_dependencies([ce0]):
            xch_seq = tensor2seq(lm.xch)
            cembed_seq = []
            for i, xch_i in enumerate(xch_seq):
                cembed_seq.append(shared_char_word_var_fm(Wc, xch_i, filtsz, char_dsz, wsz, None if i == 0 else True))
            word_char = seq2tensor(cembed_seq)

        # List to tensor, reform as (T, B, W)
        # Join embeddings along the third dimension
        joint = word_char

        inputs = tf.nn.dropout(joint, lm.pkeep)
        inputs = tf.unstack(inputs, num=lm.nbptt, axis=1)
        hsz = kwargs['hsz']
        lm._rnnlm(hsz, nlayers, inputs, vsz)
        return lm

    def get_vocab(self, vocab_type='word'):
        if vocab_type == 'char':
            return self.char_vocab
        return None

    def save_md(self, basename):
        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        if len(self.char_vocab) > 0:
            with open(basename + '-char.vocab', 'w') as f:
                json.dump(self.char_vocab, f)
        with open(basename + '-batch_dims.json', 'w') as f:
            json.dump({'batchsz': self.batchsz, 'nbptt': self.nbptt, 'maxw': self.maxw}, f)


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
