import tensorflow as tf
import numpy as np
import time
import math
from data import batch

class Trainer:

    def __init__(self, sess, model, outdir, optim, eta):
        
        self.sess = sess
        self.outdir = outdir
        self.loss = model.createLoss()
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
        
            _, step, summary_str, lossv = self.sess.run([self.train_op, self.global_step, self.summary_op, self.loss], feed_dict=feed_dict)
            self.train_writer.add_summary(summary_str, step)
        
            total_loss += lossv

        duration = time.time() - start_time
        print('Train (Loss %.4f) (%.3f sec)' % (float(total_loss)/len(ts), duration))

    def _step(self, batch):

        sentence_lengths = batch["length"]
        truth = batch["y"]
        feed_dict = self.model.ex2dict(batch, 1)
        guess = self.model.predict(self.sess, batch)
        correct_labels = 0
        total_labels = 0
        for b in range(len(guess)):
            length = sentence_lengths[b]

            assert(length == len(guess[b]))
            sentence = guess[b]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += length
        return correct_labels, total_labels

    def test(self, ts, batchsz, phase='Test'):

        total_correct = total_sum = 0
        start_time = time.time()
    
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            batch_correct, batch_total = self._step(ts_i)
            total_correct += batch_correct
            total_sum += batch_total
             
        duration = time.time() - start_time
        total_acc = total_correct / float(total_sum)
        print('%s (Acc %d/%d = %.4f) (%.3f sec)' % (phase, total_correct, total_sum, total_acc, duration))
        return total_acc
