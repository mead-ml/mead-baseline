import tensorflow as tf
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from baseline.tf.tfy import *
import tensorflow.contrib.seq2seq as tfcontrib_seq2seq
from baseline.model import EncoderDecoder, load_seq2seq_model, create_seq2seq_model


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
    def load(basename, **kwargs):

        with open(basename + '.state', 'r') as f:
            state = json.load(f)
        # FIXME: Need a single name for this.  This is a total hack
        state["layers"] = state["nlayers"]
        #print(state)
        with open(basename + '-1.vocab', 'r') as f:
            src_vocab_embed = json.load(f)

        with open(basename + '-2.vocab', 'r') as f:
            dst_vocab_embed = json.load(f)

        if 'predict' in kwargs:
            state['predict'] = kwargs['predict']

        if 'beam' in kwargs:
            state['beam'] = kwargs['beam']

        state['sess'] = kwargs.get('sess', tf.Session())

        if 'model_type' in kwargs:
            state['model_type'] = kwargs['model_type']
        elif state['attn']:
            state['model_type'] = 'attn' if state['attn'] is True else 'default'

        model = Seq2SeqModel.create(src_vocab_embed, dst_vocab_embed, **state)

        do_init = kwargs.get('init', True)
        if do_init:
            init = tf.global_variables_initializer()
            model.sess.run(init)

        model.saver = tf.train.Saver()
        model.saver.restore(model.sess, basename + '.model')
        return model

    @staticmethod
    def create(src_vocab_embed, dst_vocab_embed, **kwargs):

        model = Seq2SeqModel()

        hsz = int(kwargs['hsz'])
        attn = kwargs.get('model_type') == 'attn'
        #print('attn', attn)
        nlayers = int(kwargs.get('layers', 1))
        rnntype = kwargs.get('rnntype', 'lstm')
        mxlen = kwargs.get('mxlen', 100)
        predict = kwargs.get('predict', False)
        beam_width = kwargs.get('beam', 1)
        model.sess = kwargs.get('sess', tf.Session())
        unif = kwargs.get('unif', 0.25)
        # These are going to be (B,T)
        model.src = kwargs.get('src', tf.placeholder(tf.int32, [None, mxlen], name="src"))
        model.tgt = kwargs.get('tgt', tf.placeholder(tf.int32, [None, mxlen], name="tgt"))
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = kwargs.get('dropout', 0.5)
        model.src_len = kwargs.get('src_len', tf.placeholder(tf.int32, [None], name="src_len"))
        model.tgt_len = kwargs.get('tgt_len', tf.placeholder(tf.int32, [None], name="tgt_len"))
        model.mx_tgt_len = kwargs.get('mx_tgt_len', tf.placeholder(tf.int32, name="mx_tgt_len"))
        model.vocab1 = src_vocab_embed if type(src_vocab_embed) is dict else src_vocab_embed.vocab
        model.vocab2 = dst_vocab_embed if type(dst_vocab_embed) is dict else dst_vocab_embed.vocab

        model.mxlen = mxlen
        model.hsz = hsz
        model.nlayers = nlayers
        model.rnntype = rnntype
        model.attn = attn

        GO = model.vocab2['<GO>']
        EOS = model.vocab2['<EOS>']
        ##vsz = dst_vocab_embed.vsz + 1
        dst_vsz = len(model.vocab2)
        #if dst_vocab_embed is not None:
        #    assert dst_vsz == dst_vocab_embed.vsz + 1
        #assert src_vocab_embed.dsz == dst_vocab_embed.dsz
        model.dsz = kwargs['dsz'] if type(src_vocab_embed) is dict else src_vocab_embed.dsz

        with tf.name_scope("LUT"):
            if type(src_vocab_embed) is not dict:
                Wi = tf.Variable(tf.constant(src_vocab_embed.weights, dtype=tf.float32), name="Wi")
            else:
                Wi = tf.Variable(tf.random_uniform([len(model.vocab1), model.dsz], -unif, unif), name="Wi")
            if type(src_vocab_embed) is not dict:
                Wo = tf.Variable(tf.constant(dst_vocab_embed.weights, dtype=tf.float32), name="Wo")
            else:
                Wo = tf.Variable(tf.random_uniform([len(model.vocab2), model.dsz], -unif, unif), name="Wo")

            embed_in = tf.nn.embedding_lookup(Wi, model.src)
            
        with tf.name_scope("Recurrence"):
            rnn_enc_tensor, final_encoder_state = model.encode(embed_in, model.src)
            batch_sz = tf.shape(rnn_enc_tensor)[0]
            with tf.variable_scope("dec"):
                proj = dense_layer(dst_vsz)
                rnn_dec_cell = model._attn_cell_w_dropout(rnn_enc_tensor, beam_width) #[:,:-1,:])

                if beam_width > 1:
                    final_encoder_state = tf.contrib.seq2seq.tile_batch(final_encoder_state, multiplier=beam_width)

                if model.attn is True:
                    initial_state = rnn_dec_cell.zero_state(batch_sz*beam_width, tf.float32)
                    ###Should we pass the hidden state??
                    ###initial_state = initial_state.clone(cell_state=final_encoder_state)
                else:
                    initial_state = final_encoder_state

                if predict is True:
                    if beam_width == 1:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(Wo, tf.fill([batch_sz], GO), EOS)
                        decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper,
                                                                  initial_state=initial_state, output_layer=proj)
                    else:

                        ##initial_state = tf.contrib.seq2seq.tile_batch(initial_state, multiplier=beam_width)
                        #tiled_initial_state =
                        # Define a beam-search decoder
                        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                            cell=rnn_dec_cell,
                            embedding=Wo,
                            start_tokens=tf.fill([batch_sz], GO),
                            end_token=EOS,
                            initial_state=initial_state,
                            beam_width=beam_width,
                            output_layer=proj,
                            length_penalty_weight=0.0)
                else:
                    helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, model.tgt), sequence_length=model.tgt_len)
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper, initial_state=initial_state, output_layer=proj)

                final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                          impute_finished=predict is False or beam_width == 1,
                                                                                          swap_memory=True,
                                                                                          output_time_major=True,
                                                                                          maximum_iterations=model.mxlen)

                if predict is True and beam_width > 1:
                    model.preds = tf.no_op()
                    best = final_outputs.predicted_ids
                else:
                    model.preds = final_outputs.rnn_output
                    best = final_outputs.sample_id

        with tf.name_scope("Output"):
            model.best = tf.identity(best, name='best')
            if beam_width > 1:
                model.probs = tf.no_op(name='probs')
            else:
                model.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), model.preds)
        return model

    def _attn_cell_w_dropout(self, rnn_enc_tensor, beam):
        cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, self.nlayers)
        if self.attn:
            src_len = self.src_len
            if beam > 1:
                # Expand the encoded tensor for all beam entries
                rnn_enc_tensor = tf.contrib.seq2seq.tile_batch(rnn_enc_tensor, multiplier=beam)
                src_len = tf.contrib.seq2seq.tile_batch(src_len, multiplier=beam)
            attn_mech = tfcontrib_seq2seq.BahdanauAttention(self.hsz, rnn_enc_tensor, src_len)
            #attn_mech = tfcontrib_seq2seq.LuongAttention(self.hsz, rnn_enc_tensor, src_len)
            cell = tf.contrib.seq2seq.AttentionWrapper(cell, attn_mech, self.hsz, name='dyn_attn_cell')
        return cell

    def encode(self, embed_in, src):
        with tf.name_scope('encode'):
            # List to tensor, reform as (T, B, W)
            if self.rnntype == 'blstm':
                nlayers_bi = int(self.nlayers / 2)
                rnn_fwd_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, nlayers_bi)
                rnn_bwd_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, nlayers_bi)
                rnn_enc_tensor, final_encoder_state = tf.nn.bidirectional_dynamic_rnn(rnn_fwd_cell, rnn_bwd_cell,
                                                                                      embed_in,
                                                                                      scope='brnn_enc',
                                                                                      sequence_length=self.src_len,
                                                                                      dtype=tf.float32)
                rnn_enc_tensor = tf.concat(rnn_enc_tensor, -1)
                if nlayers_bi == 1:
                    encoder_state = final_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for i in range(self.nlayers):
                        encoder_state.append(final_encoder_state[0][i])  # forward
                        encoder_state.append(final_encoder_state[1][i])  # backward
                    encoder_state = tuple(encoder_state)
            else:

                rnn_enc_cell = multi_rnn_cell_w_dropout(self.hsz, self.pkeep, self.rnntype, self.nlayers)
                rnn_enc_tensor, encoder_state = tf.nn.dynamic_rnn(rnn_enc_cell, embed_in,
                                                                  scope='rnn_enc',
                                                                  sequence_length=self.src_len,
                                                                  dtype=tf.float32)

            # This comes out as a sequence T of (B, D)
            return rnn_enc_tensor, encoder_state

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

    def restore_graph(self, base):
        with open(base + '.graph', 'rb') as gf:
            gd = tf.GraphDef()
            gd.ParseFromString(gf.read())
            self.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

    def run(self, source_dict):
        src = source_dict['src']
        src_len = source_dict['src_len']
        feed_dict = {self.src: src, self.src_len: src_len, self.pkeep: 1.0}
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
        return self.sess.run(self.probs, feed_dict=feed_dict)

    def make_input(self, batch_dict, do_dropout=False):
        src = batch_dict['src']
        src_len = batch_dict['src_len']
        dst = batch_dict['dst']
        dst_len = batch_dict['dst_len']

        mx_tgt_len = np.max(dst_len)
        feed_dict = {self.src: src, self.src_len: src_len,
                     self.tgt: dst, self.tgt_len: dst_len,
                     self.mx_tgt_len: mx_tgt_len,
                     self.pkeep: self.pdrop_value if do_dropout else 1.0}
        return feed_dict

    def get_src_vocab(self):
        return self.vocab1

    def get_dst_vocab(self):
        return self.vocab2


BASELINE_SEQ2SEQ_MODELS = {
    'default': Seq2SeqModel.create,
    'attn': Seq2SeqModel.create
}
BASELINE_SEQ2SEQ_LOADERS = {
    'default': Seq2SeqModel.load,
    'attn': Seq2SeqModel.load
}


def create_model(src_vocab_embed, dst_vocab_embed, **kwargs):
    model = create_seq2seq_model(BASELINE_SEQ2SEQ_MODELS, src_vocab_embed, dst_vocab_embed, **kwargs)
    return model


def load_model(modelname, **kwargs):
    return load_seq2seq_model(BASELINE_SEQ2SEQ_LOADERS, modelname, **kwargs)
