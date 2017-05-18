import tensorflow as tf
import numpy as np
import time

from data import batch
import math

from utils import ProgressBar

class Trainer:

    def __init__(self, model, optim, eta, clip=5):
        
        self.loss = model.create_loss()
        self.model = model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        gvs = self.optimizer.compute_gradients(self.loss)
        #capped_gvs = [(tf.clip_by_norm(grad, -clip, clip), var) for grad, var in gvs]
        #self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def checkpoint(self, sess, outdir, name):
        self.model.saver.save(sess, outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self, sess, outdir):
        latest = tf.train.latest_checkpoint(outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def train(self, ts, sess, summary_writer, dropout, batchsz):
        total_loss = 0
        steps = int(math.floor(len(ts)/float(batchsz)))
        shuffle = np.random.permutation(np.arange(steps))
        start_time = time.time()
    
        pg = ProgressBar(steps)

        for i in range(steps):
            si = shuffle[i]
            ts_i = batch(ts, si, batchsz)
            feed_dict = self.model.ex2dict(ts_i, 1.0-dropout)
        
            _, step, summary_str, lossv = sess.run([self.train_op, self.global_step, self.summary_op, self.loss], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            #print(lossv, errv, totv)
            total_loss += lossv
            pg.update()

        pg.done()
        duration = time.time() - start_time
            
        print('Train (Loss %.4f) (%.3f sec)' % (total_loss/steps, duration))


    def test(self, ts, sess, batchsz=1):

        total_loss = 0
        steps = int(math.floor(len(ts)/float(batchsz)))
        start_time = time.time()
        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            feed_dict = self.model.ex2dict(ts_i, 1.0)
            lossv = sess.run(self.loss, feed_dict=feed_dict)
            total_loss += lossv
        
        duration = time.time() - start_time

        avg_loss = total_loss / steps
        print('Test (Loss %.4f) (%.3f sec)' % (avg_loss, duration))

        return avg_loss


    def best_in_batch(self, ts, sess, batchsz):

        steps = int(math.floor(len(ts)/float(batchsz)))
        start_time = time.time()
        shuffle = np.random.permutation(np.arange(steps))
        ts_i = batch(ts, shuffle[0], batchsz)
        feed_dict = self.model.ex2dict(ts_i, 1.0)
        best = sess.run(self.model.best, feed_dict=feed_dict)
        
        
        duration = time.time() - start_time
        print('Show (%.3f sec)' % duration)
        return best
