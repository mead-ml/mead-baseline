import tensorflow as tf
import numpy as np
import time
import math
from data import batch

class Trainer:

    def __init__(self, sess, model, outdir, optim, eta):
        
        self.sess = sess
        self.outdir = outdir
        self.loss, self.err, self.total = model.createLoss()
        self.model = model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.merge_all_summaries()
        self.train_writer = tf.summary.FileWriter(self.outdir + "/train", sess.graph)


    def writer(self):
        return self.train_writer

    def checkpoint(self, name):
        self.model.saver.save(self.sess, self.outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint(self.outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(self.sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def train(self, ts, dropout, batchsz):

        start_time = time.time()

        steps = int(math.floor(len(ts)/float(batchsz)))

        shuffle = np.random.permutation(np.arange(steps))

        total_loss = total_err = total_sum = 0

        for i in range(steps):
            si = shuffle[i]
            ts_i = batch(ts, si, batchsz)
            feed_dict = self.model.ex2dict(ts_i, 1.0-dropout)
        
            _, step, summary_str, lossv, errv, totalv = self.sess.run([self.train_op, self.global_step, self.summary_op, self.loss, self.err, self.total], feed_dict=feed_dict)
            self.train_writer.add_summary(summary_str, step)
        
            total_err += errv
            total_loss += lossv
            total_sum += totalv

        duration = time.time() - start_time
        total_correct = float(total_sum - total_err)
        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (float(total_loss)/len(ts), total_correct, total_sum, total_correct/total_sum, duration))

    def test(self, ts, batchsz, phase='Test'):

        total_loss = total_err = total_sum = 0
        start_time = time.time()
    
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            feed_dict = self.model.ex2dict(ts_i, 1)
            lossv, errv, totalv = self.sess.run([self.loss, self.err, self.total], feed_dict=feed_dict)
            total_loss += lossv
            total_err += errv
            total_sum += totalv
             
        duration = time.time() - start_time
        total_correct = float(total_sum - total_err)
        acc = total_correct / total_sum
        avg_loss = float(total_loss)/len(ts)
        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (phase, avg_loss, total_correct, total_sum, acc, duration))
        return avg_loss, acc
