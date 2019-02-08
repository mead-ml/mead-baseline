from baseline.tf.tfy import *
from baseline.version import __version__
from baseline.model import LanguageModel, register_model
from baseline.tf.embeddings import *
from baseline.tf.tfy import new_placeholder_dict, TRAIN_FLAG, lstm_cell_w_dropout
from baseline.tf.transformer import transformer_encoder_stack, subsequent_mask
from baseline.utils import read_json, write_json, ls_props
from google.protobuf import text_format
import copy
import os


class LanguageModelBase(LanguageModel):

    def __init__(self):
        self.saver = None
        self.layers = None
        self.hsz = None
        self.probs = None

    def set_saver(self, saver):
        self.saver = saver

    def decode(self, inputs, **kwargs):
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

    def _create_loss(self, scope):
        with tf.variable_scope(scope):
            vsz = self.embeddings[self.tgt_key].vsz
            targets = tf.reshape(self.y, [-1])
            bt_x_v = tf.nn.log_softmax(tf.reshape(self.logits, [-1, vsz]), axis=-1)
            one_hots = tf.one_hot(targets, vsz)
            example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
            loss = tf.reduce_mean(example_loss)
            return loss

    def create_loss(self):
        return self._create_loss(scope='loss{}'.format(self.id))

    def create_test_loss(self):
        return self._create_loss(scope='test_loss')

    def make_input(self, batch_dict, train=False):

        feed_dict = new_placeholder_dict(train)

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
        lm.id = kwargs.get('id', 0)
        lm.embeddings = embeddings

        inputs = {}
        lm.batchsz = 0
        for k, embedding in embeddings.items():
            x = kwargs.get(k, embedding.create_placeholder(name=k))
            lm.batchsz = tf.shape(x)[0]
            inputs[k] = x

        lm.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        lm.sess = kwargs.get('sess', tf.Session())
        lm.pdrop_value = kwargs.get('pdrop', 0.5)
        lm.hsz = kwargs['hsz']
        lm.tgt_key = kwargs.get('tgt_key')
        if lm.tgt_key is None:
            raise Exception('Need a `tgt_key` to know which source vocabulary should be used for destination ')
        unif = kwargs.get('unif', 0.05)
        weight_initializer = tf.random_uniform_initializer(-unif, unif)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE, initializer=weight_initializer):

            embeddings_layer = lm.embed(**kwargs)
            nc = embeddings[lm.tgt_key].vsz

            lstm_encoder_layer = lm.decode(inputs, **kwargs)
            lang_model = LangSequenceModel(nc, embeddings_layer, lstm_encoder_layer)
            lm.logits, lm.final_state = lang_model(inputs)
            lm.probs = tf.nn.softmax(lm.logits, name="softmax")

            return lm

    def embed(self, **kwargs):
        """This method performs "embedding" of the inputs.  The base method here then concatenates along depth
        dimension to form word embeddings

        :return: A 3-d vector where the last dimension is the concatenated dimensions of all embeddings
        """

        return EmbeddingsStack(self.embeddings, self.pdrop_value)


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
        vocab_b = tf.get_variable("vocab_b", [vsz],  initializer=tf.zeros_initializer(), dtype=tf.float32)
        if do_weight_tying and self.hsz == self.embeddings[self.tgt_key].get_dsz():
            with tf.variable_scope(self.embeddings[self.tgt_key].scope, reuse=True):
                W = tf.get_variable("W")
            return tf.matmul(h, W, transpose_b=True, name="logits") + vocab_b
        else:
            vocab_w = tf.get_variable(
                "vocab_w", [self.hsz, vsz], dtype=tf.float32)
            return tf.nn.xw_plus_b(h, vocab_w, vocab_b, name="logits")


@register_model(task='lm', name='default')
class RNNLanguageModel(LanguageModelBase):
    def __init__(self):
        super(RNNLanguageModel, self).__init__()
        self.rnntype = 'lstm'

    def decode(self, inputs, batchsz=1, rnntype='lstm', variational_dropout=False, **kwargs):
        lstm_encoder_layer = LSTMEncoderWithState(self.hsz, kwargs.get('layers', 1), self.pdrop_value)
        self.initial_state = lstm_encoder_layer.zero_state(self.batchsz)
        inputs["h"] = self.initial_state
        return lstm_encoder_layer
