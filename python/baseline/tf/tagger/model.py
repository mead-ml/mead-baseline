from baseline.model import TaggerModel
from baseline.tf.tfy import *
from baseline.utils import ls_props, read_json, write_json, MAGIC_VARS
from baseline.tf.embeddings import *
from baseline.tf.tfy import TRAIN_FLAG, new_placeholder_dict
from baseline.version import __version__
from baseline.model import register_model
from baseline.utils import listify, Offsets


class TaggerModelBase(TaggerModel):

    def __init__(self):
        super(TaggerModelBase, self).__init__()
        self._unserializable = []

    @property
    def lengths_key(self):
        return self._lengths_key

    @lengths_key.setter
    def lengths_key(self, value):
        self._lengths_key = value

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):
        """This method saves out a `.state` file containing meta-data from these classes and any info
        registered by a user-defined derived class as a `property`. Also write the `graph` and `saver` and `labels`

        :param basename:
        :return:
        """

        write_json(self._state, basename + '.state')
        write_json(self.labels, basename + ".labels")
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

    def _record_state(self, **kwargs):
        """
        First, write out the embedding names, so we can recover those.  Then do a deepcopy on the model init params
        so that it can be recreated later.  Anything that is a placeholder directly on this model needs to be removed

        :param kwargs:
        :return:
        """
        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__

        self._state = {k: v for k, v in kwargs.items() if k not in self._unserializable + MAGIC_VARS + list(self.embeddings.keys())}
        self._state.update({
            "version": __version__,
            "embeddings": embeddings_info
        })


    @property
    def dropin_value(self):
        return self._dropin_value

    @dropin_value.setter
    def dropin_value(self, dict_value):
        self._dropin_value = dict_value

    def drop_inputs(self, key, x, do_dropout):
        v = self.dropin_value.get(key, 0)
        if do_dropout and v > 0.0:
            drop_indices = np.where((np.random.random(x.shape) < v) & (x != Offsets.PAD))
            x[drop_indices[0], drop_indices[1]] = Offsets.UNK
        return x

    def make_input(self, batch_dict, train=False):
        feed_dict = new_placeholder_dict(train)
        for k in self.embeddings.keys():
            feed_dict["{}:0".format(k)] = self.drop_inputs(k, batch_dict[k], train)
        y = batch_dict.get('y', None)

        #feed_dict = {v.x: self.drop_inputs(k, batch_dict[k], do_dropout) for k, v in self.embeddings.items()}

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict[self.lengths] = batch_dict[self.lengths_key]

        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    @classmethod
    def load(cls, basename, **kwargs):
        """Reload the model from a graph file and a checkpoint

        The model that is loaded is independent of the pooling and stacking layers, making this class reusable
        by sub-classes.

        :param basename: The base directory to load from
        :param kwargs: See below

        :Keyword Arguments:
        * *sess* -- An optional tensorflow session.  If not passed, a new session is
            created

        :return: A restored model
        """

        _state = read_json(basename + '.state')
        _state['sess'] = kwargs.pop('sess', tf.Session())
        _state['model_type'] = kwargs.get('model_type', 'default')
        embeddings = {}
        embeddings_dict = _state.pop("embeddings")

        for key, class_name in embeddings_dict.items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            embeddings[key] = Constructor(key, **embed_args)

        labels = read_json(basename + '.labels')
        model = cls.create(embeddings, labels, **_state)
        model._state = _state
        do_init = kwargs.get('init', True)
        if do_init:
            init = tf.global_variables_initializer()
            model.sess.run(init)

        model.saver = tf.train.Saver()
        model.saver.restore(model.sess, basename)

        return model

    def save_using(self, saver):
        self.saver = saver

    def create_loss(self):
        return self.layers.neg_log_loss(self.probs, self.y, self.lengths)

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict[self.lengths_key]
        bestv = self.sess.run(self.best, feed_dict=feed_dict)
        return [pij[:sl] for pij, sl in zip(bestv, lengths)]

    def embed(self, **kwargs):
        return EmbeddingsStack(self.embeddings, self.pdrop_value)

    def encode(self, **kwargs):
        pass

    def decode(self, **kwargs):
        self.crf = bool(kwargs.get('crf', False))
        self.crf_mask = bool(kwargs.get('crf_mask', False))
        self.constraint = kwargs.get('constraint')
        if self.crf:
            return CRF(len(self.labels), self.constraint)
        return TaggerGreedyDecoder(len(self.labels), self.constraint)

    @classmethod
    def create(cls, embeddings, labels, **kwargs):

        model = cls()
        model.embeddings = embeddings
        model._record_state(**kwargs)
        model.lengths_key = kwargs.get('lengths_key')
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.labels = labels
        nc = len(labels)
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, None], name="y"))
        # This only exists to make exporting easier
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.dropin_value = kwargs.get('dropin', {})
        model.sess = kwargs.get('sess', tf.Session())
        model.pdrop_in = kwargs.get('dropin', 0.0)
        model.labels = labels
        model.span_type = kwargs.get('span_type')

        inputs = {'lengths': model.lengths}
        for k, embedding in embeddings.items():
            x = kwargs.get(k, embedding.create_placeholder(name=k))
            inputs[k] = x

        embed_model = model.embed(**kwargs)
        transduce_model = model.encode(**kwargs)
        decode_model = model.decode(**kwargs)

        model.layers = TagSequenceModel(nc, embed_model, transduce_model, decode_model)
        model.probs = model.layers.transduce(inputs)
        model.best = model.layers.decode(model.probs, model.lengths)
        return model


@register_model(task='tagger', name='default')
class RNNTaggerModel(TaggerModelBase):

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def __init__(self):
        super(RNNTaggerModel, self).__init__()

    def encode(self, **kwargs):
        self.vdrop = kwargs.get('variational_dropout', False)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])

        if rnntype == 'blstm':
            Encoder = BiLSTMEncoder
        else:
            Encoder = LSTMEncoder
        return Encoder(hsz, nlayers, self.pdrop_value, self.vdrop, rnn_signal)
            #((embedseq, lengths),
            #                                                                      training=TRAIN_FLAG())
        #return lstm_encoder(embedseq, self.lengths, hsz, self.pdrop_value, self.vdrop, rnntype, nlayers)


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(TaggerModelBase):

    def __init__(self):
        super(CNNTaggerModel, self).__init__()

    def encode(self, **kwargs):
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])
        filts = kwargs.get('wfiltsz', None)
        if filts is None:
            filts = 5

        # motsz, filtsz, activation='relu', name=None, **kwargs):
        cnnout = ParallelConvEncoder(self.layers.embed_model.dsz, hsz, listify(filts))
        return cnnout
