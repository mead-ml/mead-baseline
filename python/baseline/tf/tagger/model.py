import os
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.contrib.layers import fully_connected, xavier_initializer
from baseline.model import TaggerModel, create_tagger_model, load_tagger_model
from baseline.tf.tfy import *
from baseline.utils import ls_props, read_json, write_json
from baseline.tf.embeddings import *
from baseline.version import __version__
from baseline.model import register_model
from baseline.utils import listify, Offsets


class TaggerModelBase(TaggerModel):

    @property
    def lengths_key(self):
        return self._lengths_key

    @lengths_key.setter
    def lengths_key(self, value):
        self._lengths_key = value

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        # For each embedding, save a record of the keys

        embeddings_info = {}
        for k, v in self.embeddings.items():
            embeddings_info[k] = v.__class__.__name__
        state = {
            'version': __version__,
            'embeddings': embeddings_info,
            'crf': self.crf,
            'proj': self.proj,
            'constrain_decode': True if self.constraint is not None else False
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        write_json(self.labels, basename + ".labels")
        for key, embedding in self.embeddings.items():
            embedding.save_md(basename + '-{}-md.json'.format(key))
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

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
        * *session* -- An optional tensorflow session.  If not passed, a new session is
            created

        :return: A restored model
        """
        sess = kwargs.get('session', kwargs.get('sess', tf.Session()))
        model = cls()
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename

        state = read_json(basename + '.state')
        for prop in ls_props(model):
            if prop in state:
                setattr(model, prop, state[prop])

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            try:
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            except:
                # Backwards compat
                sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name + ".model"})

        model.embeddings = dict()
        for key, class_name in state['embeddings'].items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            embed_args[key] = tf.get_default_graph().get_tensor_by_name('{}:0'.format(key))
            Constructor = eval(class_name)
            model.embeddings[key] = Constructor(key, **embed_args)

        model.crf = bool(state.get('crf', False))
        model.proj = bool(state.get('proj', False))
        model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')

        model.y = tf.get_default_graph().get_tensor_by_name('y:0')
        model.probs = tf.get_default_graph().get_tensor_by_name('output/probs:0')
        model.best = tf.get_default_graph().get_tensor_by_name('output/best:0')

        try:
            model.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
            if not model.crf:
                print('Warning: meta-data says no CRF but model contains transition matrix!')
                model.crf = True
        except:
            if model.crf is True:
                print('Warning: meta-data says there is a CRF but not transition matrix found!')
            model.A = None
            model.crf = False

        model.labels = read_json(basename + '.labels')
        model.sess = sess
        model.saver = tf.train.Saver(saver_def=saver_def)
        return model

    def save_using(self, saver):
        self.saver = saver

    def _compute_word_level_loss(self, mask):

        nc = len(self.labels)
        # Cross entropy loss
        cross_entropy = tf.one_hot(self.y, nc, axis=-1) * tf.log(tf.nn.softmax(self.probs))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, axis=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def _compute_sentence_level_loss(self):

        if self.constraint is not None:
            A = tf.get_variable(
                "transitions_raw",
                shape=(len(self.labels), len(self.labels)),
                dtype=tf.float32,
                trainable=True
            )

            self.mask, inv_mask = self.constraint
            self.inv_mask = inv_mask * tf.constant(-1e4)

            self.A = tf.add(tf.multiply(A, self.mask), self.inv_mask, name="transitions")
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths, self.A)
        else:
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths)
        return tf.reduce_mean(-ll)

    def _create_sentence_level_decode(self, trans, norm=False):
        bsz = tf.shape(self.probs)[0]
        lsz = len(self.labels)
        np_gos = np.full((1, 1, lsz), -1e4, dtype=np.float32)
        np_gos[:, :, Offsets.GO] = 0
        gos = tf.constant(np_gos)
        start = tf.tile(gos, [bsz, 1, 1])
        start = tf.nn.log_softmax(start, axis=-1) if norm else start
        probv = tf.concat([start, self.probs], axis=1)
        viterbi, _ = tf.contrib.crf.crf_decode(probv, trans, self.lengths + 1)
        self.best = tf.identity(viterbi[:, 1:], name="best")

    def _create_word_level_decode(self):
        self.best = tf.argmax(self.probs, 2, name="best")

    def create_loss(self):

        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float32)
            mask = tf.sign(gold)

            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss()
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        with tf.variable_scope(self.out_scope, auxiliary_name_scope=False) as s:
            with tf.name_scope(s.original_name_scope):
                if self.crf:
                    self._create_sentence_level_decode(self.A)
                else:
                    if self.constraint is not None:
                        self.constraint = tf.nn.log_softmax(self.constraint[1] * tf.constant(-1e4), axis=-1)
                        self._create_sentence_level_decode(self.constraint, norm=True)
                    else:
                        self._create_word_level_decode()

        return all_loss

    def __init__(self):
        super(TaggerModelBase, self).__init__()

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict[self.lengths_key]
        bestv = self.sess.run(self.best, feed_dict=feed_dict)
        return [pij[:sl] for pij, sl in zip(bestv, lengths)]

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
        return tf.layers.dropout(word_embeddings, self.pdrop_value, training=TRAIN_FLAG())

    def encode(self, embedseq, **kwargs):
        pass

    @classmethod
    def create(cls, embeddings, labels, **kwargs):

        model = cls()
        model.embeddings = embeddings
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
        model.crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.proj = bool(kwargs.get('proj', False))
        model.feed_input = bool(kwargs.get('feed_input', False))
        model.activation_type = kwargs.get('activation', 'tanh')
        model.constraint = kwargs.get('constraint')

        embedseq = model.embed()
        seed = np.random.randint(10e8)
        enc_out = model.encode(embedseq, **kwargs)

        with tf.variable_scope("output") as model.out_scope:
            if model.feed_input is True:
                enc_out = tf.concat(axis=2, values=[enc_out, embedseq])

            # Converts seq to tensor, back to (B,T,W)
            T = tf.shape(enc_out)[1]
            H = enc_out.get_shape()[2]
            # Flatten from [B x T x H] - > [BT x H]
            enc_out_bt_x_h = tf.reshape(enc_out, [-1, H])
            init = xavier_initializer(True, seed)

            with tf.contrib.slim.arg_scope([fully_connected], weights_initializer=init):
                if model.proj is True:
                    hidden = tf.layers.dropout(fully_connected(enc_out_bt_x_h, H,
                                                           activation_fn=tf_activation(model.activation_type)), model.pdrop_value, training=TRAIN_FLAG())
                    preds = fully_connected(hidden, nc, activation_fn=None, weights_initializer=init)
                else:
                    preds = fully_connected(enc_out_bt_x_h, nc, activation_fn=None, weights_initializer=init)
            model.probs = tf.reshape(preds, [-1, T, nc], name="probs")
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

    def encode(self, embedseq, **kwargs):
        self.vdrop = kwargs.get('variational_dropout', False)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])
        if rnntype == 'blstm':
            rnnfwd = stacked_lstm(hsz//2, self.pdrop_value, nlayers, self.vdrop, training=TRAIN_FLAG())
            rnnbwd = stacked_lstm(hsz//2, self.pdrop_value, nlayers, self.vdrop, training=TRAIN_FLAG())
            rnnout, _ = tf.nn.bidirectional_dynamic_rnn(rnnfwd, rnnbwd, embedseq, sequence_length=self.lengths, dtype=tf.float32)
            # The output of the BRNN function needs to be joined on the H axis
            rnnout = tf.concat(axis=2, values=rnnout)
        else:
            rnnfwd = stacked_lstm(hsz, self.pdrop_value, nlayers, self.vdrop, training=TRAIN_FLAG())
            rnnout, _ = tf.nn.dynamic_rnn(rnnfwd, embedseq, sequence_length=self.lengths, dtype=tf.float32)
        return rnnout


@register_model(task='tagger', name='cnn')
class CNNTaggerModel(TaggerModelBase):

    def __init__(self):
        super(CNNTaggerModel, self).__init__()

    def encode(self, embedseq, **kwargs):
        nlayers = kwargs.get('layers', 1)
        hsz = int(kwargs['hsz'])
        filts = kwargs.get('wfiltsz', None)
        if filts is None:
            filts = 5

        cnnout = stacked_cnn(embedseq, hsz, self.pdrop_value, nlayers, filts=listify(filts), training=TRAIN_FLAG())
        return cnnout
