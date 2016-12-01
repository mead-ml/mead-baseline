import tensorflow as tf
import numpy as np
import time

class Trainer:

    def __init__(self, model, optim, eta):
        
        self.loss, self.acc = model.createLoss()
        self.model = model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.acc_summary = tf.scalar_summary("accuracy", self.acc)
        self.summary_op = tf.merge_all_summaries()

    def checkpoint(self, sess, outdir, name):
        self.model.saver.save(sess, outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self, sess, outdir):
        latest = tf.train.latest_checkpoint(outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def train(self, ts, sess, summary_writer, dropout):

        total_loss = total_corr = total = 0
        seq = np.random.permutation(len(ts))
        start_time = time.time()
        for j in seq:
    
            feed_dict = self.model.ex2dict(ts[j], 1.0-dropout)
        
            _, step, summary_str, lossv, accv = sess.run([self.train_op, self.global_step, self.summary_op, self.loss, self.acc], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
        
            total_corr += accv
            total_loss += lossv
            total += ts[j]["y"].shape[0]

        duration = time.time() - start_time

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))

    def test(self, ts, sess, phase='Test'):

        total_loss = total_corr = total = 0
        start_time = time.time()
        
        for j in range(len(ts)):
            
            feed_dict = self.model.ex2dict(ts[j], 1)
            lossv, accv = sess.run([self.loss, self.acc], feed_dict=feed_dict)
            total_corr += accv
            total_loss += lossv
            total += ts[j]["y"].shape[0]
        
        duration = time.time() - start_time

        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (phase, float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
        return float(total_corr)/total
