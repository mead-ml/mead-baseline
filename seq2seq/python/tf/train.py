import tensorflow as tf
import numpy as np
import time

import math

from utils import ProgressBar, lookup_sentence

class Trainer:

    def __init__(self, sess, model, optim, eta, mom, clip, pdrop):
        self.sess = sess
        self.loss = model.create_loss()
        self.model = model
        self.pdrop = pdrop
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            if mom > 0:
                self.optimizer = tf.train.MomentumOptimizer(eta, mom)
            else:
                self.optimizer = tf.train.GradientDescentOptimizer(eta)

        gvs = self.optimizer.compute_gradients(self.loss)
        #capped_gvs = [(tf.clip_by_norm(grad, -clip, clip), var) for grad, var in gvs]
        #self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def checkpoint(self, outdir, name):
        self.model.saver.save(self.sess, outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(outdir):
        latest = tf.train.latest_checkpoint(outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(self.sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def step(self, src, dst, pkeep=1.0):
        """
        Generate probability distribution over output V for next token
        """

        return self.sess.run(model.probs, feed_dict=feed_dict)

    def train(self, ts):
        total_loss = 0
        steps = len(ts)    
        pg = ProgressBar(steps)
        for src,tgt,src_len,tgt_len in ts:
            mx_tgt_len = np.max(tgt_len)
            feed_dict = {self.model.src: src, self.model.tgt: tgt, self.model.src_len: src_len, self.model.tgt_len: tgt_len, self.model.mx_tgt_len: mx_tgt_len,self.model.pkeep: 1-self.pdrop}
            _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            pg.update()

        pg.done()
        avg_loss = total_loss/steps
        return avg_loss

    def test(self, ts):

        total_loss = 0
        steps = len(ts)
        for src,tgt,src_len,tgt_len in ts:
            mx_tgt_len = np.max(tgt_len)
            feed_dict = {self.model.src: src, self.model.tgt: tgt, self.model.src_len: src_len, self.model.tgt_len: tgt_len, self.model.mx_tgt_len: mx_tgt_len, self.model.pkeep: 1}

            lossv = self.sess.run(self.loss, feed_dict=feed_dict)
            total_loss += lossv

        avg_loss = total_loss / steps
        return avg_loss


def show_examples(sess, model, es, rlut1, rlut2, embed2, mxlen, sample, prob_clip, max_examples, reverse):
    si = np.random.randint(0, len(es))

    src_array, tgt_array, src_len, _ = es[si]

    if max_examples > 0:
        max_examples = min(max_examples, src_array.shape[0])
        src_array = src_array[0:max_examples]
        tgt_array = tgt_array[0:max_examples]
        src_len = src_len[0:max_examples]

    GO = embed2.vocab['<GO>']
    EOS = embed2.vocab['<EOS>']
    
    for src_len_i,src_i,tgt_i in zip(src_len, src_array, tgt_array):

        print('========================================================================')

        sent = lookup_sentence(rlut1, src_i, reverse=reverse)
        print('[OP] %s' % sent)
        sent = lookup_sentence(rlut2, tgt_i)
        print('[Actual] %s' % sent)
        dst_i = np.zeros((1, mxlen))
        src_i = src_i[np.newaxis,:]
        src_len_i = np.array([src_len_i])
        next_value = GO
        for j in range(mxlen):
            dst_i[0, j] = next_value
            tgt_len_i = np.array([j+1])
            #output = model.step(sess, src_i, src_len_i, dst_i, tgt_len_i, j)[0]
            output = model.step(sess, src_i, src_len_i, dst_i, tgt_len_i)[j]
            if sample is False:
                next_value = np.argmax(output)
            else:
                # This is going to zero out low prob. events so they are not
                # sampled from
                next_value = beam_multinomial(prob_clip, output)

            if next_value == EOS:
                break

        sent = lookup_sentence(rlut2, dst_i.squeeze())
        print('Guess: %s' % sent)
        print('------------------------------------------------------------------------')


def fit(sess, seq2seq, ts, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    gpu = bool(kwargs['gpu']) if 'gpu' in kwargs else True
    optim = kwargs['optim'] if 'optim' in kwargs else 'adam'
    eta = float(kwargs['eta']) if 'eta' in kwargs else 0.01
    mom = float(kwargs['mom']) if 'mom' in kwargs else 0.9
    clip = float(kwargs['clip']) if 'clip' in kwargs else 5
    model_file = kwargs['outfile'] if 'outfile' in kwargs and kwargs['outfile'] is not None else './model.pyth'
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    pdrop = kwargs['pdrop'] if 'pdrop' in kwargs else 0.5
    trainer = Trainer(sess, seq2seq, optim, eta, mom, clip, pdrop)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    trainer.prepare(saver)

    val_min = 1000
    last_improved = 0

    for i in range(epochs):
        print('Training epoch %d' % (i+1))
        start_time = time.time()
        avg_train_loss = trainer.train(ts)
        duration = time.time() - start_time
        print('Training Loss %.4f (Perplexity %.4f) (%.3f sec)' % 
              (avg_train_loss, np.exp(avg_train_loss), duration))


        if after_train_fn is not None:
            after_train_fn(seq2seq)

        start_time = time.time()
        avg_val_loss = trainer.test(es)
        duration = time.time() - start_time
        print('Validation Loss %.4f (Perplexity %.4f) (%.3f sec)' % 
              (avg_val_loss, np.exp(avg_val_loss), duration))

        if avg_val_loss < val_min:
            last_improved = i
            val_min = avg_val_loss
            print('Lowest error achieved yet -- writing model')
            seq2seq.save(sess, model_file)

        if (i - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

