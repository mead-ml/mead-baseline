import tensorflow as tf
import numpy as np
import time
import math
from data import batch
from utils import toSpans, fScore
class Trainer:

    def __init__(self, sess, model, outdir, optim, eta, idx2label, fscore=0):
        
        self.sess = sess
        self.outdir = outdir
        self.loss = model.createLoss()
        self.model = model
        self.idx2label = idx2label
        self.fscore = fscore
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.MomentumOptimizer(eta, 0.9)

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

        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0
        
        for b in range(len(guess)):
            length = sentence_lengths[b]
            assert(length == len(guess[b]))
            sentence = guess[b]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += length

            if self.fscore > 0:
                gold_chunks = toSpans(gold, self.idx2label)
                gold_count += len(gold_chunks)

                guess_chunks = toSpans(sentence, self.idx2label)
                guess_count += len(guess_chunks)
            
                overlap_chunks = gold_chunks & guess_chunks
                overlap_count += len(overlap_chunks)

        return correct_labels, total_labels, overlap_count, gold_count, guess_count

    def test(self, ts, batchsz, phase='Test'):

        total_correct = total_sum = fscore = 0
        total_gold_count = total_guess_count = total_overlap_count = 0
        start_time = time.time()
    
        steps = int(math.floor(len(ts)/float(batchsz)))

        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            correct, count, overlaps, golds, guesses = self._step(ts_i)
            total_correct += correct
            total_sum += count
            total_gold_count += golds
            total_guess_count += guesses
            total_overlap_count += overlaps

        duration = time.time() - start_time
        total_acc = total_correct / float(total_sum)

        # Only show the fscore if requested
        if self.fscore > 0:
            fscore = fScore(total_overlap_count,
                            total_gold_count,
                            total_guess_count,
                            self.fscore)
            print('%s (F%d = %.4f) (Acc %d/%d = %.4f) (%.3f sec)' % 
                  (phase,
                   self.fscore,
                   fscore,
                   total_correct,
                   total_sum,
                   total_acc,
                   duration))
                        
        else:
            print('%s (Acc %d/%d = %.4f) (%.3f sec)' %
                  (phase,
                   total_correct,
                   total_sum, total_acc,
                   duration))

        return total_acc, fscore
