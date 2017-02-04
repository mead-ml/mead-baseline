import tensorflow as tf
import numpy as np
import json
from google.protobuf import text_format
from tensorflow.python.platform import gfile
import math
from utils import *


# Method for seq2seq w/ attention using TF's library
def attn_rnn_seq2seq(encoder_inputs,
                     decoder_inputs,
                     cell,
                     num_heads=1,
                     dtype=tf.float32,
                     scope=None):
    with tf.variable_scope(scope or "attention_rnn_seq2seq"):
        encoder_outputs, enc_state = tf.contrib.rnn.static_rnn(cell, encoder_inputs, dtype=dtype)
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                      for e in encoder_outputs]
        attention_states = tf.concat(values=top_states, axis=1)
    
    return tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs,
                                                       enc_state,
                                                       attention_states,
                                                       cell,
                                                       num_heads=num_heads)

def tensor2seq(tensor):
    return tf.unstack(tf.transpose(tensor, perm=[1, 0, 2]))

def seq2tensor(sequence):
    return tf.transpose(tf.stack(sequence), perm=[1, 0, 2])

    
class Seq2SeqBase:

    def makeCell(self, hsz, nlayers, rnntype):
        st = False if nlayers > 1 else True
        
        cell = tf.contrib.rnn.BasicLSTMCell(hsz, state_is_tuple=st) if rnntype == 'lstm' else tf.contrib.rnn.GRUCell(hsz)
        
        if nlayers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * nlayers, state_is_tuple=st)
        return cell

    def save(self, sess, outdir, base):
        basename = outdir + '/' + base
        tf.train.write_graph(sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w+b') as f:
            f.write(str(self.saver.as_saver_def()))
        self.saver.save(sess, basename + '.model')

        with open(basename + '-1.vocab', 'w') as f:
            json.dump(self.vocab1, f)      

        with open(basename + '-2.vocab', 'w') as f:
            json.dump(self.vocab2, f)      

    def fmtfor(self, i):
        if i == 0:
            return 'output/probs:0'
        else:
            return 'output/probs_%d:0' % i

    def restore(self, sess, indir, base, maxlen):
        basename = indir + '/' + base
        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'r') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(gd, name='')
            print('restore_op_name %s' % saver_def.restore_op_name)
            print('filename_tensor_name %s' % saver_def.filename_tensor_name)
            sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: basename + '.model'})
            self.src = tf.get_default_graph().get_tensor_by_name('src:0')
            self.dst = tf.get_default_graph().get_tensor_by_name('dst:0')
            self.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            self.probs = [tf.get_default_graph().get_tensor_by_name(self.fmtfor(i)) for i in range(maxlen)]

        with open(basename + '-1.vocab', 'r') as f:
            self.vocab1 = json.load(f)

        with open(basename + '-2.vocab', 'r') as f:
            self.vocab2 = json.load(f)

            
        self.saver = tf.train.Saver(saver_def=saver_def)


    def __init__(self):
        pass

    def ex2dict(self, example, pkeep):
        return {self.src: example["src"], 
                self.dst: example["dst"], 
                self.tgt: example["tgt"], 
                self.pkeep: pkeep}

    def createLoss(self):

        tsparse = tf.unstack(tf.transpose(self.tgt, perm=[1, 0]))

        with tf.name_scope("Loss"):

            log_perp_list = []
            error_list = []
            total_list = []
            # For each t in T
            for preds_i, target_i in zip(self.preds, tsparse):

                # Mask against (B)
                mask = tf.cast(tf.sign(target_i), tf.float32)
                # self.preds_i = (B, V)
                best_i = tf.cast(tf.argmax(preds_i, 1), tf.int32)
                err = tf.cast(tf.not_equal(best_i, target_i), tf.float32)
                # Gives back (B, V)
                xe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=preds_i, labels=target_i)

                log_perp_list.append(xe * mask)
                error_list.append(err * mask)
                total_list.append(tf.reduce_sum(mask))
                
            log_perps = tf.add_n(log_perp_list)
            error_all = tf.add_n(error_list)
            totalsz = tf.add_n(total_list)
            log_perps /= totalsz

            cost = tf.reduce_sum(log_perps)
            all_error = tf.reduce_sum(error_all)

            batchsz = tf.cast(tf.shape(tsparse[0])[0], tf.float32)
            avg_cost = cost/batchsz
            #checks = tf.add_check_numerics_ops()
            return avg_cost, all_error, totalsz #, checks

    def step(self, sess, src, dst):
        """
        Generate probability distribution over output V for next token
        """
        feed_dict = {self.src: src, self.dst: dst, self.pkeep: 1.0}
        return sess.run(self.probs, feed_dict=feed_dict)


class Seq2SeqModel(Seq2SeqBase):

    def __init__(self):
        pass

    def params(self, embed1, embed2, maxlen, hsz, nlayers=1, attn=False, rnntype='lstm'):
        # These are going to be (B,T)
        self.src = tf.placeholder(tf.int32, [None, maxlen], name="src")
        self.dst = tf.placeholder(tf.int32, [None, maxlen], name="dst")
        self.tgt = tf.placeholder(tf.int32, [None, maxlen], name="tgt")
        self.vocab1 = embed1.vocab
        self.vocab2 = embed2.vocab
        # ADDME!
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name = "W")
            Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name = "W")

            ei0 = tf.scatter_update(Wi, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
            eo0 = tf.scatter_update(Wo, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        
            with tf.control_dependencies([ei0]):
                embed_in = tf.nn.embedding_lookup(Wi, self.src)


            with tf.control_dependencies([eo0]):
                embed_out = tf.nn.embedding_lookup(Wo, self.dst)

        with tf.name_scope("Recurrence"):
            # List to tensor, reform as (T, B, W)
            embed_in_seq = tensor2seq(embed_in)
            embed_out_seq = tensor2seq(embed_out)

            rnn_enc = self.makeCell(hsz, nlayers, rnntype)
            rnn_dec = self.makeCell(hsz, nlayers, rnntype)

            # Primitive will wrap RNN and unroll in time
            rnn_enc_seq, final_encoder_state = tf.contrib.rnn.static_rnn(rnn_enc, embed_in_seq, scope='rnn_enc', dtype=tf.float32)
            # Provides the link between the encoder final state and the decoder
            rnn_dec_seq, _ = tf.contrib.rnn.static_rnn(rnn_dec, embed_out_seq, initial_state=final_encoder_state, scope='rnn_dec', dtype=tf.float32)

        with tf.name_scope("output"):
            # Leave as a sequence of (T, B, W)

            W = tf.Variable(tf.truncated_normal([hsz, embed2.vsz + 1],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1, embed2.vsz + 1]), name="b")

            self.preds = [(tf.matmul(rnn_dec_i, W) + b) for rnn_dec_i in rnn_dec_seq]
            self.probs = [tf.nn.softmax(pred, name="probs") for pred in self.preds]



class Seq2SeqLib(Seq2SeqBase):

    def __init__(self):
        pass

    def params(self, embed1, embed2, maxlen, hsz, nlayers=1, attn=False, rnntype='lstm'):
        # These are going to be (B,T)
        self.src = tf.placeholder(tf.int32, [None, maxlen], name="src")
        self.dst = tf.placeholder(tf.int32, [None, maxlen], name="dst")
        self.tgt = tf.placeholder(tf.int32, [None, maxlen], name="tgt")
        self.vocab1 = embed1.vocab
        self.vocab2 = embed2.vocab
        # ADDME!
        self.pkeep = tf.placeholder(tf.float32, name="pkeep")

        with tf.name_scope("LUT"):
            Wi = tf.Variable(tf.constant(embed1.weights, dtype=tf.float32), name = "W")
            Wo = tf.Variable(tf.constant(embed2.weights, dtype=tf.float32), name = "W")

            ei0 = tf.scatter_update(Wi, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
            eo0 = tf.scatter_update(Wo, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, embed1.dsz]))
        
            with tf.control_dependencies([ei0]):
                embed_in = tf.nn.embedding_lookup(Wi, self.src)


            with tf.control_dependencies([eo0]):
                embed_out = tf.nn.embedding_lookup(Wo, self.dst)

        with tf.name_scope("Recurrence"):
            embed_in_seq = tensor2seq(embed_in)
            embed_out_seq = tensor2seq(embed_out)

            cell = self.makeCell(hsz, nlayers, rnntype)
            if attn:
                print('With attention')
                rnn_dec_seq, _ = attn_rnn_seq2seq(embed_in_seq, embed_out_seq, cell)
            else:
                rnn_dec_seq, _ = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(embed_in_seq, embed_out_seq, cell)
            
        with tf.name_scope("output"):
            # Leave as a sequence of (T, B, W)

            W = tf.Variable(tf.truncated_normal([hsz, embed2.vsz + 1],
                                                stddev = 0.1), name="W")
            b = tf.Variable(tf.constant(0.0, shape=[1, embed2.vsz + 1]), name="b")

            self.preds = [(tf.matmul(rnn_dec_i, W) + b) for rnn_dec_i in rnn_dec_seq]
            self.probs = [tf.nn.softmax(pred, name="probs") for pred in self.preds]
