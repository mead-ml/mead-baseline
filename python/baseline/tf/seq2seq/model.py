import tensorflow as tf
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from baseline.tf.tfy import *
from baseline.w2v import RandomInitVecModel
import tensorflow.contrib.seq2seq as tfcontrib_seq2seq
from baseline.model import EncoderDecoder


def _temporal_cross_entropy_loss(logits, labels, label_lengths, mx_seq_length):
    """Do cross-entropy loss accounting for sequence lengths
    
    :param logits: a `Tensor` with shape `[batch, timesteps, vocab]`
    :param labels: an integer `Tensor` with shape `[batch, timesteps]`
    :param label_lengths: The actual length of the target text.  Assume right-padded
    :param mx_seq_length: The maximum length of the sequence
    :return: 
    """

    # The labels actual length is 100, and starts with <GO>
    labels = tf.transpose(labels, perm=[1, 0])
    # TxB loss mask
    # Logits == mxlen=10*batchsz=100
    # Labels == mxlen=9,batchsz=100
    labels = labels[0:mx_seq_length, :]
    timesteps = tf.to_int32(tf.shape(labels)[0])
    # The labels no longer include <GO> so go is not useful.  This means that if the length was 100 before, the length
    # of labels is now 99 (and that is the max allowed)

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


class Seq2SeqModel(EncoderDecoder):

    def create_loss(self):
        # We do not want to count <GO> in our assessment, we do want to count <EOS>
        return _temporal_cross_entropy_loss(self.preds[:-1, :, :], self.tgt[:, 1:], self.tgt_len - 1, self.mx_tgt_len - 1)

    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        pass

    @staticmethod
    def create(src_vocab_embed, dst_vocab_embed, **kwargs):

        model = Seq2SeqModel()
        hsz = int(kwargs['hsz'])
        attn = bool(kwargs.get('attn', False))
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'lstm')
        mxlen = kwargs.get('mxlen', 100)
        predict = kwargs.get('predict', False)
        model.sess = kwargs.get('sess', tf.Session())

        # These are going to be (B,T)
        model.src = tf.placeholder(tf.int32, [None, mxlen], name="src")
        model.tgt = tf.placeholder(tf.int32, [None, mxlen], name="tgt")
        model.pkeep = tf.placeholder(tf.float32, name="pkeep")

        model.src_len = tf.placeholder(tf.int32, [None], name="src_len")
        model.tgt_len = tf.placeholder(tf.int32, [None], name="tgt_len")
        model.mx_tgt_len = tf.placeholder(tf.int32, name="mx_tgt_len")

        model.vocab1 = src_vocab_embed.vocab
        model.vocab2 = dst_vocab_embed.vocab

        model.mxlen = mxlen
        model.hsz = hsz
        model.nlayers = nlayers
        model.rnntype = rnntype
        model.attn = attn

        GO = model.vocab2['<GO>']
        EOS = model.vocab2['<EOS>']
        vsz = dst_vocab_embed.vsz + 1

        assert src_vocab_embed.dsz == dst_vocab_embed.dsz
        model.dsz = src_vocab_embed.dsz

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(src_vocab_embed.weights, dtype=tf.float32), name="Wi")
            Wo = tf.Variable(tf.constant(dst_vocab_embed.weights, dtype=tf.float32), name="Wo")

            embed_in = tf.nn.embedding_lookup(Wi, model.src)
            
        with tf.name_scope("Recurrence"):
            rnn_enc_tensor, final_encoder_state = model.encode(embed_in, model.src)
            batch_sz = tf.shape(rnn_enc_tensor)[0]

            with tf.variable_scope("dec"):
                proj = dense_layer(vsz)
                rnn_dec_cell = model._attn_cell(rnn_enc_tensor) #[:,:-1,:])

                if model.attn is True:
                    initial_state = rnn_dec_cell.zero_state(dtype=tf.float32, batch_size=batch_sz)
                else:
                    initial_state = final_encoder_state

                if predict is True:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(Wo, tf.fill([batch_sz], GO), EOS)
                else:
                    helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, model.tgt), sequence_length=model.tgt_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper, initial_state=initial_state, output_layer=proj)
                final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, output_time_major=True, maximum_iterations=model.mxlen)
                model.preds = final_outputs.rnn_output
                best = final_outputs.sample_id

        with tf.name_scope("Output"):
            model.best = tf.identity(best, name='best')
            model.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), model.preds)
        return model

    def _attn_cell(self, rnn_enc_tensor):
        cell = new_multi_rnn_cell(self.hsz, self.rnntype, self.nlayers)
        if self.attn:
            attn_mech = tfcontrib_seq2seq.BahdanauAttention(self.hsz, rnn_enc_tensor, self.src_len)
            #attn_mech = tfcontrib_seq2seq.LuongAttention(self.hsz, rnn_enc_tensor, self.src_len)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
        return cell

    def encode(self, embed_in, src):
        with tf.name_scope('encode'):
            # List to tensor, reform as (T, B, W)
            embed_in_seq = tensor2seq(embed_in)
            rnn_enc_cell = new_multi_rnn_cell(self.hsz, self.rnntype, self.nlayers)
            #TODO: Switch to tf.nn.rnn.dynamic_rnn()
            rnn_enc_seq, final_encoder_state = tf.contrib.rnn.static_rnn(rnn_enc_cell, embed_in_seq, scope='rnn_enc', dtype=tf.float32)
            # This comes out as a sequence T of (B, D)
            return seq2tensor(rnn_enc_seq), final_encoder_state

    def save_md(self, model_base):

        path_and_file = model_base.split('/')
        outdir = '/'.join(path_and_file[:-1])
        base = path_and_file[-1]
        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)

        state = {"attn": self.attn, "hsz": self.hsz, "dsz": self.dsz, "rnntype": self.rnntype, "nlayers": self.nlayers, "mxlen": self.mxlen }
        with open(model_base + '.state', 'w') as f:
            json.dump(state, f)

        with open(model_base + '-1.vocab', 'w') as f:
            json.dump(self.vocab1, f)      

        with open(model_base + '-2.vocab', 'w') as f:
            json.dump(self.vocab2, f)     

    def save(self, model_base):
        self.save_md(model_base)
        self.saver.save(self.sess, model_base + '.model')

    def restore_md(self, model_base):

        with open(model_base + '-1.vocab', 'r') as f:
            self.vocab1 = json.load(f)

        with open(model_base + '-2.vocab', 'r') as f:
            self.vocab2 = json.load(f)

        with open(model_base + '.state', 'r') as f:
            state = json.load(f)
            self.attn = state['attn']
            self.hsz = state['hsz']
            self.dsz = state['dsz']
            self.rnntype = state['rnntype']
            self.nlayers = state['nlayers']
            self.mxlen = state['mxlen']

    def restore_graph(self, base):
        with open(base + '.graph', 'rb') as gf:
            gd = tf.GraphDef()
            gd.ParseFromString(gf.read())
            self.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

    def step(self, src, src_len, dst, dst_len):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = self.make_feed_dict(src, src_len, dst, dst_len)
        return self.sess.run(self.probs, feed_dict=feed_dict)

    def make_feed_dict(self, src, src_len, dst, dst_len, do_dropout=False):
        mx_tgt_len = np.max(dst_len)
        feed_dict = {self.src: src, self.src_len: src_len, self.tgt: dst, self.tgt_len: dst_len, self.mx_tgt_len: mx_tgt_len, self.pkeep: 1.0}
        return feed_dict

    def get_src_vocab(self):
        return self.vocab1

    def get_dst_vocab(self):
        return self.vocab2


def create_model(src_vocab_embed, dst_vocab_embed, **kwargs):
    enc_dec = Seq2SeqModel.create(src_vocab_embed, dst_vocab_embed, **kwargs)
    return enc_dec
