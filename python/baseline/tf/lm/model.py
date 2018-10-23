from baseline.tf.tfy import *
from baseline.version import __version__
from baseline.model import LanguageModel, register_model
from baseline.tf.embeddings import *
from baseline.utils import read_json, write_json, ls_props
from google.protobuf import text_format


class LanguageModelBase(LanguageModel):

    def __init__(self):
        self.pkeep = None
        self.saver = None
        self.layers = None
        self.hsz = None
        self.probs = None

    def save_using(self, saver):
        self.saver = saver

    def decode(self, inputs):
        pass

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__
        state = {
            "version": __version__,
            "embeddings": embeddings_info,
            "hsz": self.hsz,
            "layers": self.layers,
            "tgt_key": self.tgt_key
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        for key, embedding in self.embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

        write_json(state, basename + '.state')
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

    def create_loss(self):
        with tf.variable_scope("Loss"):
            targets = tf.reshape(self.y, [-1])
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones([tf.size(targets)], dtype=tf.float32)])
            loss = tf.reduce_sum(loss) / self.batchsz
            return loss

    def make_input(self, batch_dict, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.embeddings.keys():

            feed_dict["{}:0".format(key)] = batch_dict[key]

        y = batch_dict.get('y')
        if y is not None:
            feed_dict[self.y] = batch_dict['y']

        return feed_dict

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        step_softmax = self.sess.run(self.probs, feed_dict)
        return step_softmax

    @classmethod
    def create(cls, embeddings, **kwargs):

        lm = cls()
        lm.embeddings = embeddings
        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        lm.batchsz = kwargs['batchsz']
        lm.sess = kwargs.get('sess', tf.Session())
        lm.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        pdrop = kwargs.get('pdrop', 0.5)
        lm.pdrop_value = pdrop
        lm.hsz = kwargs['hsz']
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')
        unif = kwargs.get('unif', 0.05)
        weight_initializer = tf.random_uniform_initializer(-unif, unif)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE, initializer=weight_initializer):

            inputs = lm.embed()
            lm.layers = kwargs.get('layers', 1)
            h = lm.decode(inputs, **kwargs)
            lm.logits = lm.output(h, embeddings[lm.tgt_key].vsz, **kwargs)
            lm.probs = tf.nn.softmax(lm.logits, name="softmax")

            return lm

    def embed(self):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """
        all_embeddings_out = []
        for embedding in self.embeddings.values():
            embeddings_out = embedding.encode()
            all_embeddings_out.append(embeddings_out)
        word_embeddings = tf.concat(values=all_embeddings_out, axis=2)
        return tf.nn.dropout(word_embeddings, self.pkeep)

    @classmethod
    def load(cls, basename, **kwargs):
        state = read_json(basename + '.state')
        if 'predict' in kwargs:
            state['predict'] = kwargs['predict']

        if 'sampling' in kwargs:
            state['sampling'] = kwargs['sampling']

        if 'sampling_temp' in kwargs:
            state['sampling_temp'] = kwargs['sampling_temp']

        if 'beam' in kwargs:
            state['beam'] = kwargs['beam']

        if 'batchsz' in kwargs:
            state['batchsz'] = kwargs['batchsz']

        state['sess'] = kwargs.get('sess', tf.Session())

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        embeddings = dict()
        embeddings_dict = state.pop('embeddings')
        for key, class_name in embeddings_dict.items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            embeddings[key] = Constructor(key, **embed_args)

        model = cls.create(embeddings, **state)
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        do_init = kwargs.get('init', True)
        if do_init:
            init = tf.global_variables_initializer()
            model.sess.run(init)

        model.saver = tf.train.Saver()
        model.saver.restore(model.sess, basename)
        return model

    def output(self, h, vsz, **kwargs):
        # Do weight sharing if we can
        do_weight_tying = bool(kwargs.get('tie_weights', False))
        if do_weight_tying and self.hsz == self.embeddings[self.tgt_key].get_dsz():
            with tf.variable_scope(self.embeddings[self.tgt_key].scope, reuse=True):
                W = tf.get_variable("W")
            return tf.matmul(h, W, transpose_b=True, name="logits")
        else:
            vocab_w = tf.get_variable(
                "vocab_w", [self.hsz, vsz], dtype=tf.float32)
            vocab_b = tf.get_variable("vocab_b", [vsz], dtype=tf.float32)

            return tf.nn.xw_plus_b(h, vocab_w, vocab_b, name="logits")


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):
    def __init__(self):
        super(RNNLanguageModel, self).__init__()
        self.rnntype = 'lstm'

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def decode(self, inputs, **kwargs):

        def cell():
            return lstm_cell_w_dropout(self.hsz, self.pkeep, variational=self.vdrop)

        self.rnntype = kwargs.get('rnntype', 'lstm')
        self.vdrop = kwargs.get('variational_dropout', False)

        rnnfwd = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.layers)], state_is_tuple=True)
        self.initial_state = rnnfwd.zero_state(self.batchsz, tf.float32)
        rnnout, state = tf.nn.dynamic_rnn(rnnfwd, inputs, initial_state=self.initial_state, dtype=tf.float32)
        h = tf.reshape(tf.concat(rnnout, 1), [-1, self.hsz])
        self.final_state = state
        return h

