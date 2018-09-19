from baseline.tf.tfy import *
from baseline.model import create_lang_model
from baseline.tf.embeddings import *
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

        def cell():
            return lstm_cell_w_dropout(self.hsz, self.pkeep)
        rnnfwd = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.layers)], state_is_tuple=True)

        self.initial_state = rnnfwd.zero_state(self.batchsz, tf.float32)
        rnnout, state = tf.nn.dynamic_rnn(rnnfwd, inputs, initial_state=self.initial_state, dtype=tf.float32)
        output = tf.reshape(tf.concat(rnnout, 1), [-1, self.hsz])

        softmax_w = tf.get_variable(
            "softmax_w", [self.hsz, vsz], dtype=tf.float32)
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

    def get_vocab(self, vocab_type='word'):
        pass


class WordLanguageModel(AbstractLanguageModel):

    def __init__(self):
        AbstractLanguageModel.__init__(self)

    def make_input(self, batch_dict, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        y = batch_dict.get('y')
        if y is not None:
            feed_dict[self.y] = batch_dict['y']

        return feed_dict

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()

        lm.embeddings = dict()
        for key in embeddings.keys():
            DefaultType = TensorFlowCharConvEmbeddings if key == 'char' else TensorFlowTokenEmbeddings
            lm.embeddings[key] = tf_embeddings(embeddings[key], key, DefaultType=DefaultType, **kwargs)
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        lm.batchsz = kwargs['batchsz']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.rnntype = kwargs.get('rnntype', 'lstm')
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        pdrop = kwargs.get('pdrop', 0.5)
        lm.pdrop_value = pdrop
        lm.hsz = kwargs['hsz']

        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            if 'x' in lm.tgt_key:
                lm.tgt_key = 'x'
            elif 'word' in lm.tgt_key:
                lm.tgt_key = 'word'
        unif = kwargs.get('unif', 0.05)
        weight_initializer = tf.random_uniform_initializer(-unif, unif)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE, initializer=weight_initializer):

            all_embeddings_out = []
            for embedding in lm.embeddings.values():
                embeddings_out = embedding.encode()
                all_embeddings_out += [embeddings_out]

            word_embeddings = tf.concat(values=all_embeddings_out, axis=2)
            inputs = tf.nn.dropout(word_embeddings, lm.pkeep)
            lm.layers = kwargs.get('layers', kwargs.get('nlayers', 1))
            lm._rnnlm(inputs, len(embeddings[lm.tgt_key].vocab))
            return lm

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {'hsz': self.hsz, 'batchsz': self.batchsz, 'layers': self.layers}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        #if len(self.word_vocab) > 0:
        #    with open(basename + '-word.vocab', 'w') as f:
        #        json.dump(self.word_vocab, f)




BASELINE_LM_MODELS = {
    'default': WordLanguageModel.create
}

# TODO:
# BASELINE_LM_LOADERS = {
#    'default': WordLanguageModel.load,
#    'convchar': CharCompLanguageModel.load
# }


# TODO: move the scoping and weight initialization into the model itself
def create_model(embeddings, **kwargs):

    if 'sess' not in kwargs:
        kwargs['sess'] = tf.Session()

    lm = create_lang_model(BASELINE_LM_MODELS, embeddings, **kwargs)
    return lm
