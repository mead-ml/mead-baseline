import tensorflow as tf
import numpy as np
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from distutils.version import LooseVersion
import math
from utils import *
from tfy import *
from w2v import *

class Seq2SeqBase:

    def save(self, sess, model_base):
        pass

    def __init__(self):
        pass

    def step(self, sess, src, src_len, dst, dst_len):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = {self.src: src, self.tgt: dst, self.pkeep: pkeep}
        return sess.run(self.probs, feed_dict=feed_dict)


    def make_cell(self, hsz, nlayers, rnntype):
        
        if nlayers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([new_rnn_cell(hsz, rnntype, True) for _ in range(nlayers)], state_is_tuple=True)
            return cell
        return new_rnn_cell(hsz, rnntype, False)

    def create_loss(self):
        
        targets = tf.unstack(tf.transpose(self.tgt[:,1:], perm=[1, 0]))
        predictions = tf.unstack(self.preds)
        bests = tf.unstack(self.best)

        with tf.name_scope("Loss"):

            log_perp_list = []
            total_list = []
            # For each t in T
            for preds_i, best_i, target_i in zip(predictions, bests, targets):
                # Mask against (B)
                mask = tf.cast(tf.sign(target_i), tf.float32)
                # self.preds_i = (B, V)
                #best_i = tf.cast(tf.argmax(preds_i, 1), tf.int32)
                err = tf.cast(tf.not_equal(tf.cast(best_i, tf.int32), target_i), tf.float32)
                # Gives back (B, V)
                xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_i, labels=target_i)

                log_perp_list.append(xe * mask)
                total_list.append(tf.reduce_sum(mask))
                
            log_perps = tf.add_n(log_perp_list)
            totalsz = tf.add_n(total_list)
            log_perps /= totalsz

            cost = tf.reduce_sum(log_perps)

            batchsz = tf.cast(tf.shape(targets[0])[0], tf.float32)
            avg_cost = cost/batchsz
            return avg_cost

class Seq2SeqBase_v1_0(Seq2SeqBase):
    def save(self, sess, model_base):

        path_and_file = model_base.split('/')
        outdir = '/'.join(path_and_file[:-1])
        base = path_and_file[-1]
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)

        with open(model_base + '.saver', 'w+b') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(sess, model_base + '.model')

        with open(model_base + '-1.vocab', 'w') as f:
            json.dump(self.vocab1, f)      

        with open(model_base + '-2.vocab', 'w') as f:
            json.dump(self.vocab2, f)      

    def restore(self, sess, model_base, mxlen):
        with open(model_base + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(model_base + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            print('restore_op_name %s' % saver_def.restore_op_name)
            print('filename_tensor_name %s' % saver_def.filename_tensor_name)
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: model_base + '.model'})

            self.src = tf.get_default_graph().get_tensor_by_name('src:0')
            self.tgt = tf.get_default_graph().get_tensor_by_name('tgt:0')
            self.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            self.probs = [tf.get_default_graph().get_tensor_by_name(self._fmtfor(i)) for i in range(mxlen)]

        with open(model_base + '-1.vocab', 'r') as f:
            self.vocab1 = json.load(f)

        with open(model_base + '-2.vocab', 'r') as f:
            self.vocab2 = json.load(f)

            
        self.saver = tf.train.Saver(saver_def=saver_def)

    def _fmtfor(self, i):
        if i == 0:
            return 'output/probs:0'
        else:
            return 'output/probs_%d:0' % i

class Seq2SeqModel_v1_0(Seq2SeqBase_v1_0):

    def __init__(self):
        pass

    def params(self, embed1, embed2, mxlen, hsz, nlayers=1, attn=False, rnntype='lstm'):
        # These are going to be (B,T)
        self.src = tf.placeholder(tf.int32, [None, mxlen], name="src")
        self.tgt = tf.placeholder(tf.int32, [None, mxlen], name="tgt")
        self.vocab1 = embed1.vocab
        self.vocab2 = embed2.vocab
        # ADDME!
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name = "Wi")
            Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name = "Wo")

            ei0 = tf.scatter_update(Wi, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
            eo0 = tf.scatter_update(Wo, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        
            with tf.control_dependencies([ei0]):
                embed_in = tf.nn.embedding_lookup(Wi, self.src)


            with tf.control_dependencies([eo0]):
                embed_out = tf.nn.embedding_lookup(Wo, self.tgt)

        with tf.name_scope("Recurrence"):
            # List to tensor, reform as (T, B, W)
            embed_in_seq = tensor2seq(embed_in)
            embed_out_seq = tensor2seq(embed_out)

            with tf.variable_scope('var_rnn_enc'):
                rnn_enc = self.make_cell(hsz, nlayers, rnntype)
            with tf.variable_scope('var_rnn_dec'):
                rnn_dec = self.make_cell(hsz, nlayers, rnntype)
            rnn_enc_seq, final_encoder_state = tf.contrib.rnn.static_rnn(rnn_enc, embed_in_seq, scope='rnn_enc', dtype=tf.float32)

            # Provides the link between the encoder final state and the decoder
            rnn_dec_seq, _ = tf.contrib.rnn.static_rnn(rnn_dec, embed_out_seq, initial_state=final_encoder_state, scope='rnn_dec', dtype=tf.float32)

        with tf.name_scope("output"):
            # Leave as a sequence of (T, B, W)

            W = tf.Variable(tf.truncated_normal([hsz, embed2.vsz + 1],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1, embed2.vsz + 1]), name="b")

            self.preds = [(tf.matmul(rnn_dec_i, W) + b) for rnn_dec_i in rnn_dec_seq]
            self.best = tf.stack([tf.argmax(pred, 1, name='best') for pred in self.preds])
            self.probs = tf.stack([tf.nn.softmax(pred, name="probs") for pred in self.preds])
            self.preds = tf.stack(self.preds)


class LegacySeq2SeqLib(Seq2SeqBase_v1_0):

    def __init__(self):
        pass

    def params(self, embed1, embed2, mxlen, hsz, nlayers=1, attn=False, rnntype='lstm'):
        # These are going to be (B,T)
        self.src = tf.placeholder(tf.int32, [None, mxlen], name="src")
        self.tgt = tf.placeholder(tf.int32, [None, mxlen], name="tgt")
        self.vocab1 = embed1.vocab
        self.vocab2 = embed2.vocab
        # ADDME!
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name = "Wi")
            Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name = "Wo")

            ei0 = tf.scatter_update(Wi, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
            eo0 = tf.scatter_update(Wo, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        
            with tf.control_dependencies([ei0]):
                embed_in = tf.nn.embedding_lookup(Wi, self.src)


            with tf.control_dependencies([eo0]):
                embed_out = tf.nn.embedding_lookup(Wo, self.tgt)

        with tf.name_scope("Recurrence"):
            embed_in_seq = tensor2seq(embed_in)
            embed_out_seq = tensor2seq(embed_out)

            cell = self.make_cell(hsz, nlayers, rnntype)
            if attn:
                print('With attention')
                rnn_dec_seq, _ = legacy_attn_rnn_seq2seq(embed_in_seq, embed_out_seq, cell)
            else:
                rnn_dec_seq, _ = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(embed_in_seq, embed_out_seq, cell)
            
        with tf.name_scope("output"):
            # Leave as a sequence of (T, B, W)

            W = tf.Variable(tf.truncated_normal([hsz, embed2.vsz + 1],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1, embed2.vsz + 1]), name="b")

            self.preds = [(tf.matmul(rnn_dec_i, W) + b) for rnn_dec_i in rnn_dec_seq]
            self.best = tf.stack([tf.argmax(pred, 1, name='best') for pred in self.preds])
            self.probs = tf.stack([tf.nn.softmax(pred, name="probs") for pred in self.preds])
            self.preds = tf.stack(self.preds)

class Seq2SeqModel_v1_1(Seq2SeqBase):

    def create_loss(self):

        targets = tf.transpose(self.tgt[:,1:], perm=[1, 0])
        targets = targets[0:self.mx_tgt_len,:]
        target_lens = self.tgt_len - 1
        with tf.name_scope("Loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.preds, labels=targets)

            loss_mask = tf.sequence_mask(
                tf.to_int32(target_lens), tf.to_int32(tf.shape(targets)[0]))
            losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])
    
            losses = tf.reduce_sum(losses)
            losses /= tf.cast(tf.reduce_sum(target_lens), tf.float32)
            return losses

    def __init__(self):
        pass

    def params(self, embed1, embed2, mxlen, hsz, nlayers=1, attn=False, rnntype='lstm', predict=False):

        # These are going to be (B,T)
        self.src = tf.placeholder(tf.int32, [None, mxlen], name="src")
        self.tgt = tf.placeholder(tf.int32, [None, mxlen], name="tgt")
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")

        self.src_len = tf.placeholder(tf.int32, [None], name="src_len")
        self.tgt_len = tf.placeholder(tf.int32, [None], name="tgt_len")
        self.mx_tgt_len = tf.placeholder(tf.int32, name="mx_tgt_len")

        self.vocab1 = embed1.vocab
        self.vocab2 = embed2.vocab

        self.mxlen = mxlen
        self.hsz = hsz
        self.nlayers = nlayers
        self.rnntype = rnntype
        self.attn = attn

        GO = self.vocab2['<GO>']
        EOS = self.vocab2['<EOS>']
        vsz = embed2.vsz + 1

        assert embed1.dsz == embed2.dsz
        self.dsz = embed1.dsz

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name="Wi")
            Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name="Wo")

            embed_in = tf.nn.embedding_lookup(Wi, self.src)
            
        with tf.name_scope("Recurrence"):
            rnn_enc_tensor, final_encoder_state = self.encode(embed_in, self.src)
            #print(final_encoder_state[0], final_encoder_state[1])
            batch_sz = tf.shape(rnn_enc_tensor)[0]

            with tf.variable_scope("dec") as vs:
                proj = dense_layer(vsz)
                rnn_dec_cell = self._attn_cell(rnn_enc_tensor) #[:,:-1,:])

                if self.attn is True:
                    initial_state = rnn_dec_cell.zero_state(dtype=tf.float32, batch_size=batch_sz)
                else:
                    initial_state = final_encoder_state

                if predict is True:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(Wo, tf.fill([batch_sz], GO), EOS)
                else:
                    helper = tf.contrib.seq2seq.TrainingHelper(inputs=tf.nn.embedding_lookup(Wo, self.tgt), sequence_length=self.tgt_len)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=rnn_dec_cell, helper=helper, initial_state=initial_state, output_layer=proj)
                final_outputs, final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, output_time_major=True, maximum_iterations=self.mxlen)
                self.preds = final_outputs.rnn_output
                best = final_outputs.sample_id

        with tf.name_scope("Output"):
            self.best = tf.identity(best, name='best')
            self.probs = tf.map_fn(lambda x: tf.nn.softmax(x, name='probs'), self.preds)
        return self

    def _attn_cell(self, rnn_enc_tensor):
        cell = new_multi_rnn_cell(self.hsz, self.rnntype, self.nlayers)
        if self.attn:
            attn_mech = tf.contrib.seq2seq.LuongAttention(self.hsz, rnn_enc_tensor, self.src_len) 
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

    def save_md(self, sess, model_base):

        path_and_file = model_base.split('/')
        outdir = '/'.join(path_and_file[:-1])
        base = path_and_file[-1]
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)

        state = {"attn": self.attn, "hsz": self.hsz, "dsz": self.dsz, "rnntype": self.rnntype, "nlayers": self.nlayers, "mxlen": self.mxlen }
        with open(model_base + '.state', 'w') as f:
            json.dump(state, f)

        with open(model_base + '-1.vocab', 'w') as f:
            json.dump(self.vocab1, f)      

        with open(model_base + '-2.vocab', 'w') as f:
            json.dump(self.vocab2, f)     
        

    def save(self, sess, model_base):
        self.save_md(sess, model_base)
        self.saver.save(sess, model_base + '.model')

    def restore_md(self, sess, model_base):

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

    def restore_graph(self, sess, indir, base):
        with open(indir + '/' + base + '.graph', 'rb') as gf:
            gd = tf.GraphDef()
            gd.ParseFromString(gf.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')

    def step(self, sess, src, src_len, dst, dst_len):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = {self.src: src, self.src_len: src_len, self.tgt: dst, self.tgt_len: dst_len, self.pkeep: 1.0}
        return sess.run(self.probs, feed_dict=feed_dict)

class Seq2Seq:

    @staticmethod
    def from_file(sess, indir, base, predict=False):        
        seq2seq = Seq2SeqModel_v1_1()
        seq2seq.restore_md(sess, indir, base)
        embed1 = RandomInitVecModel(seq2seq.dsz, seq2seq.vocab1, False)
        embed2 = RandomInitVecModel(seq2seq.dsz, seq2seq.vocab2, False)
        return seq2seq.params(embed1, embed2, seq2seq.mxlen, seq2seq.hsz, seq2seq.nlayers, seq2seq.attn, seq2seq.rnntype, predict)

    @staticmethod
    def create_lstm_attn(embed1, embed2, mxlen, hsz, layers, predict=False):

        if TF_GTE_11:
            enc_dec = Seq2SeqModel_v1_1()
        else:
            enc_dec = LegacySeq2SeqLib()
        print(enc_dec)
        return enc_dec.params(embed1, embed2, mxlen, hsz, layers, True, 'lstm', predict)

    @staticmethod
    def create_gru_attn(embed1, embed2, mxlen, hsz, layers, predict=False):

        if TF_GTE_11:
            enc_dec = Seq2SeqModel_v1_1()
        else:
            enc_dec = LegacySeq2SeqLib()
        print(enc_dec)
        return enc_dec.params(embed1, embed2, mxlen, hsz, layers, True, 'gru', predict)

    @staticmethod
    def create_lstm(embed1, embed2, mxlen, hsz, layers, predict=False):

        if TF_GTE_11:
            enc_dec = Seq2SeqModel_v1_1()
        else:
            enc_dec = Seq2SeqModel_v1_0()
        print(enc_dec)
        return enc_dec.params(embed1, embed2, mxlen, hsz, layers, False, 'lstm', predict)
    
    @staticmethod
    def create_gru(embed1, embed2, mxlen, hsz, layers, predict=False):

        if TF_GTE_11:
            enc_dec = Seq2SeqModel_v1_1()
        else:
            enc_dec = Seq2SeqModel_v1_0()
        print(enc_dec)
        return enc_dec.params(embed1, embed2, mxlen, hsz, layers, False, 'gru', predict)

    


