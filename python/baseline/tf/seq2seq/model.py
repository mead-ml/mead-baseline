from baseline.tf.seq2seq.encoders import RNNEncoder, TransformerEncoder
from baseline.tf.seq2seq.decoders import RNNDecoder, RNNDecoderWithAttn, TransformerDecoder
from google.protobuf import text_format
from baseline.tf.tfy import *
from baseline.model import EncoderDecoderModel, register_model
from baseline.utils import ls_props, read_json
from baseline.tf.embeddings import *
from baseline.version import __version__
import copy


def _temporal_cross_entropy_loss(logits, labels, label_lengths, mx_seq_length):
    """Do cross-entropy loss accounting for sequence lengths
    
    :param logits: a `Tensor` with shape `[timesteps, batch, timesteps, vocab]`
    :param labels: an integer `Tensor` with shape `[batch, timesteps]`
    :param label_lengths: The actual length of the target text.  Assume right-padded
    :param mx_seq_length: The maximum length of the sequence
    :return: 
    """

    # The labels actual length is 100, and starts with <GO>
    labels = tf.transpose(labels, perm=[1, 0])
    # TxB loss mask
    labels = labels[0:mx_seq_length, :]
    logit_length = tf.to_int32(tf.shape(logits)[0])
    timesteps = tf.to_int32(tf.shape(labels)[0])
    # The labels no longer include <GO> so go is not useful.  This means that if the length was 100 before, the length
    # of labels is now 99 (and that is the max allowed)
    pad_size = timesteps - logit_length
    logits = tf.pad(logits, [[0, pad_size], [0, 0], [0, 0]])
    #logits = logits[0:mx_seq_length, :, :]
    with tf.name_scope("Loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        # BxT loss mask
        loss_mask = tf.to_float(tf.sequence_mask(tf.to_int32(label_lengths), timesteps))
        # TxB losses * TxB loss_mask
        losses = losses * tf.transpose(loss_mask, [1, 0])

        losses = tf.reduce_sum(losses)
        losses /= tf.cast(tf.reduce_sum(label_lengths), tf.float32)
        return losses


class Seq2SeqParallelModel(EncoderDecoderModel):

    def __init__(self, create_fn, src_embeddings, tgt_embedding, **kwargs):
        super(Seq2SeqParallelModel, self).__init__()
        # We need to remove these because we may be calling back to our caller, and we need
        # the condition of calling to be non-parallel
        gpus = kwargs.pop('gpus', -1)
        # If the gpu ID is set to -1, use CUDA_VISIBLE_DEVICES to figure it out
        if gpus == -1:
            gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))
        print('Num GPUs', gpus)

        self.saver = None
        self.replicas = []
        self.parallel_params = dict()
        split_operations = dict()
        for key in src_embeddings.keys():
            EmbeddingType = src_embeddings[key].__class__
            self.parallel_params[key] = kwargs.get(key, EmbeddingType.create_placeholder('{}_parallel'.format(key)))
            split_operations[key] = tf.split(self.parallel_params[key], gpus)

        EmbeddingType = tgt_embedding.__class__
        self.parallel_params['tgt'] = kwargs.get(key, EmbeddingType.create_placeholder('tgt_parallel'.format(key)))
        split_operations['tgt'] = tf.split(self.parallel_params[key], gpus)

        self.src_lengths_key = kwargs.get('src_lengths_key')
        self.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len_parallel"))
        src_len_splits = tf.split(self.src_len, gpus)
        split_operations['src_len'] = src_len_splits

        self.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len_parallel"))
        tgt_len_splits = tf.split(self.tgt_len, gpus)
        split_operations['tgt_len'] = tgt_len_splits

        self.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))
        self.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, (), name="pkeep"))
        self.pdrop_value = kwargs.get('dropout', 0.5)

        losses = []
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        with tf.device(tf.DeviceSpec(device_type="CPU")):
            self.inference = create_fn(src_embeddings, tgt_embedding, sess=sess, mx_tgt_len=self.mx_tgt_len, pkeep=self.pkeep, id=1, **kwargs)
        for i in range(gpus):
            with tf.device(tf.DeviceSpec(device_type='GPU', device_index=i)):

                kwargs_single = copy.deepcopy(kwargs)
                kwargs_single['sess'] = sess
                kwargs_single['pkeep'] = self.pkeep
                kwargs_single['id'] = i + 1
                for k, split_operation in split_operations.items():
                    kwargs_single[k] = split_operation[i]
                replica = create_fn(src_embeddings, tgt_embedding, **kwargs_single)
                self.replicas.append(replica)
                loss_op = replica.create_loss()
                losses.append(loss_op)

        self.loss = tf.reduce_mean(tf.stack(losses))

        self.sess = sess
        self.best = self.inference.best

    def create_loss(self):
        return self.loss

    def create_test_loss(self):
        return self.inference.create_test_loss()

    def save(self, model_base):
        return self.inference.save(model_base)

    def set_saver(self, saver):
        self.inference.saver = saver
        self.saver = saver

    def step(self, batch_dict):
        """
        Generate probability distribution over output V for next token
        """
        return self.inference.step(batch_dict)

    def make_input(self, batch_dict, do_dropout=False):
        if do_dropout is False:
            return self.inference.make_input(batch_dict)

        tgt = batch_dict.get['tgt']
        tgt_len = batch_dict['tgt_len']
        mx_tgt_len = np.max(tgt_len)
        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep, "tgt:0": tgt, self.tgt_len: tgt_len, self.mx_tgt_len: mx_tgt_len}

        for key in self.parallel_params.keys():
            feed_dict["{}_parallel:0".format(key)] = batch_dict[key]

        # Allow us to track a length, which is needed for BLSTMs
        feed_dict[self.src_len] = batch_dict[self.src_lengths_key]
        return feed_dict

    def load(self, basename, **kwargs):
        self.inference.load(basename, **kwargs)


class EncoderDecoderModelBase(EncoderDecoderModel):

    def create_loss(self):
        with tf.variable_scope('Loss{}'.format(self.id), reuse=False):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def create_test_loss(self):
        with tf.variable_scope('Loss', reuse=False):
            # We do not want to count <GO> in our assessment, we do want to count <EOS>
            return _temporal_cross_entropy_loss(self.decoder.preds[:-1, :, :], self.tgt_embedding.x[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def __init__(self):
        super(EncoderDecoderModelBase, self).__init__()
        self.saver = None

    @classmethod
    def load(cls, basename, **kwargs):
        state = read_json(basename + '.state')
        if 'predict' in kwargs:
            state['predict'] = kwargs['predict']

        if 'beam' in kwargs:
            state['beam'] = kwargs['beam']

        state['sess'] = kwargs.get('sess', tf.Session())

        if 'model_type' in kwargs:
            state['model_type'] = kwargs['model_type']
        elif state['attn']:
            print('setting to attn')
            state['model_type'] = 'attn' if state['attn'] is True else 'default'

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        src_embeddings = dict()
        src_embeddings_dict = state.pop('src_embeddings')
        for key, class_name in src_embeddings_dict.items():
            md = read_json('{}-{}-md.json'.format(basename, key))
            embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
            Constructor = eval(class_name)
            src_embeddings[key] = Constructor(key, **embed_args)

        tgt_class_name = state.pop('tgt_embedding')
        md = read_json('{}-tgt-md.json'.format(basename))
        embed_args = dict({'vsz': md['vsz'], 'dsz': md['dsz']})
        Constructor = eval(tgt_class_name)
        tgt_embedding = Constructor('tgt', **embed_args)
        model = cls.create(src_embeddings, tgt_embedding, **state)
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

    def embed(self):
        all_embeddings_src = []
        for embedding in self.src_embeddings.values():
            embeddings_out = embedding.encode()
            all_embeddings_src.append(embeddings_out)

        embed_in = tf.concat(values=all_embeddings_src, axis=-1)
        return embed_in

    @classmethod
    def create(cls, src_embeddings, tgt_embedding, **kwargs):
        gpus = kwargs.get('gpus')
        if gpus is not None:
            return Seq2SeqParallelModel(cls.create, src_embeddings, tgt_embedding, **kwargs)
        model = cls()
        model.src_embeddings = src_embeddings
        model.tgt_embedding = tgt_embedding
        model.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len"))
        model.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len"))
        model.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))
        model.src_lengths_key = kwargs.get('src_lengths_key')
        model.id = kwargs.get('id', 0)
        model.sess = kwargs.get('sess', tf.Session())
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.pkeep = kwargs.get('pkeep', tf.placeholder_with_default(1.0, shape=(), name="pkeep"))

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            embed_in = model.embed()
            encoder_output = model.encode(embed_in, **kwargs)
            model.decode(encoder_output, **kwargs)
            # writer = tf.summary.FileWriter('blah', model.sess.graph)
            return model

    @property
    def src_lengths_key(self):
        return self._src_lengths_key

    @src_lengths_key.setter
    def src_lengths_key(self, value):
        self._src_lengths_key = value

    def set_saver(self, saver):
        self.saver = saver

    @property
    def src_lengths_key(self):
        return self._src_lengths_key

    @src_lengths_key.setter
    def src_lengths_key(self, value):
        self._src_lengths_key = value

    def create_encoder(self):
        pass

    def create_decoder(self, tgt_embedding, **kwargs):
        pass

    def decode(self, encoder_output, **kwargs):
        self.decoder = self.create_decoder(self.tgt_embedding, **kwargs)
        predict = kwargs.get('predict', False)
        if predict:
            self.decoder.predict(encoder_output, self.src_len, self.pkeep, **kwargs)
        else:
            self.decoder.decode(encoder_output, self.src_len, self.tgt_len, self.pkeep, **kwargs)

    def encode(self, embed_in, **kwargs):
        with tf.variable_scope('encode'):
            self.encoder = self.create_encoder()
            return self.encoder.encode(embed_in, self.src_len, self.pkeep, **kwargs)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])
        src_embeddings_info = {}
        for k, v in self.src_embeddings.items():
            src_embeddings_info[k] = v.__class__.__name__
        state = {
            "version": __version__,
            "src_embeddings": src_embeddings_info,
            "tgt_embedding": self.tgt_embedding.__class__.__name__
        }
        for prop in ls_props(self):
            state[prop] = getattr(self, prop)

        write_json(state, basename + '.state')
        for key, embedding in self.src_embeddings.items():
            embedding.save_md('{}-{}-md.json'.format(basename, key))

        self.tgt_embedding.save_md('{}-tgt-md.json'.format(basename))

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

    def save(self, model_base):
        ##self.save_md(model_base)
        self.saver.save(self.sess, model_base)

    def restore_graph(self, base):
        with open(base + '.graph', 'rb') as gf:
            gd = tf.GraphDef()
            gd.ParseFromString(gf.read())
            self.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

    def predict(self, batch_dict):
        feed_dict = self.make_input(batch_dict)
        vec = self.sess.run(self.best, feed_dict=feed_dict)
        # (B x K x T)
        if len(vec.shape) == 3:
            return vec.transpose(1, 2, 0)
        else:
            return vec.transpose(1, 0)

    def step(self, batch_dict):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = self.make_input(batch_dict)
        x = self.sess.run(self.decoder.probs, feed_dict=feed_dict)
        return x

    def make_input(self, batch_dict, do_dropout=False):

        pkeep = 1.0 - self.pdrop_value if do_dropout else 1.0
        feed_dict = {self.pkeep: pkeep}

        for key in self.src_embeddings.keys():
            feed_dict["{}:0".format(key)] = batch_dict[key]

        if self.src_lengths_key is not None:
            feed_dict[self.src_len] = batch_dict[self.src_lengths_key]

        tgt = batch_dict.get('tgt')
        if tgt is not None:
            feed_dict["tgt:0"] = batch_dict['tgt']
            feed_dict[self.tgt_len] = batch_dict['tgt_lengths']
            feed_dict[self.mx_tgt_len] = np.max(batch_dict['tgt_lengths'])

        return feed_dict


@register_model(task='seq2seq', name=['default', 'attn'])
class Seq2Seq(EncoderDecoderModelBase):

    def __init__(self):
        super(Seq2Seq, self).__init__()

    @property
    def vdrop(self):
        return self._vdrop

    @vdrop.setter
    def vdrop(self, value):
        self._vdrop = value

    def create_encoder(self):
        return RNNEncoder()

    def create_decoder(self, tgt_embedding, **kwargs):
        attn = kwargs.get('model_type') == 'attn'
        if attn:
            return RNNDecoderWithAttn(tgt_embedding, **kwargs)
        return RNNDecoder(tgt_embedding, **kwargs)


@register_model(task='seq2seq', name='transformer')
class TransformerModel(EncoderDecoderModelBase):
    def __init__(self):
        super(TransformerModel, self).__init__()

    def create_encoder(self):
        return TransformerEncoder()

    def create_decoder(self, tgt_embedding, **kwargs):
        return TransformerDecoder(tgt_embedding, **kwargs)
