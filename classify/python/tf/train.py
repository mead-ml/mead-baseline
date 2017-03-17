import tensorflow as tf
import numpy as np
import time
import math
from os import sys, path
import data
from utils import ProgressBar

class Trainer:

    def __init__(self, sess, model, outdir, optim, eta):
        
        self.sess = sess
        self.loss = model.create_loss()
        self.model = model
        self.outdir = outdir
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.outdir + "/train", sess.graph)

    def writer(self):
        return self.train_writer

    def checkpoint(self, name):
        self.model.saver.save(self.sess, self.outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint(self.outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(self.sess, latest)

    def train(self, ts, cm, dropout, batchsz=1):

        total_loss = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))
        shuffle = np.random.permutation(np.arange(steps))
        pg = ProgressBar(steps)
        cm.reset()

        for i in range(steps):
            si = shuffle[i]
            ts_i = data.batch(ts, si, batchsz)
            feed_dict = self.model.ex2dict(ts_i, 1.0-dropout)
        
            _, step, summary_str, lossv, guess = self.sess.run([self.train_op, self.global_step, self.summary_op, self.loss, self.model.best], feed_dict=feed_dict)
            self.train_writer.add_summary(summary_str, step)
            total_loss += lossv
            cm.add_batch(ts_i.y, guess)
            pg.update()

        pg.done()
        total = cm.get_total()
        total_corr = cm.get_correct()
        duration = time.time() - start_time

        print('Train (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
        print(cm)

    def test(self, ts, cm, batchsz=1, phase='Test'):

        total_loss = 0
        start_time = time.time()
        steps = int(math.floor(len(ts)/float(batchsz)))
        cm.reset()
        for i in range(steps):
            
            ts_i = data.batch(ts, i, batchsz)
            
            feed_dict = self.model.ex2dict(ts_i, 1)
            lossv, guess = self.sess.run([self.loss, self.model.best], feed_dict=feed_dict)
            cm.add_batch(ts_i.y, guess)
            total_loss += lossv

        total = cm.get_total()
        total_corr = cm.get_correct()
        
        duration = time.time() - start_time
        print('%s (Loss %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % (phase, float(total_loss)/total, total_corr, total, float(total_corr)/total, duration))
        print(cm)

        return float(total_corr)/total
