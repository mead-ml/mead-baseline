import tensorflow as tf
import numpy as np
import time
import math
from data import batch
from utils import to_spans, f_score

class Evaluator:
    
    def __init__(self, sess, model, idx2label, fscore):
        self.sess = sess
        self.model = model
        self.idx2label = idx2label
        self.fscore = fscore

    def _write_sentence_conll(self, handle, sentence, gold, txt):

        if len(txt) != len(sentence):
            txt = txt[:len(sentence)]

        try:
            for word, truth, guess in zip(txt, gold, sentence):
                handle.write('%s %s %s\n' % (word, self.idx2label[truth], self.idx2label[guess]))
            handle.write('\n')
        except:
            print('ERROR: Failed to write lines... closing file')
            handle.close()
            handle = None

    def _batch(self, batch, handle=None, txts=None):

        sentence_lengths = batch["length"]
        truth = batch["y"]
        guess = self.model.predict(self.sess, batch)
        correct_labels = 0
        total_labels = 0

        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0
        
        # For each sentence
        for b in range(len(guess)):
            length = sentence_lengths[b]
            assert(length == len(guess[b]))
            sentence = guess[b]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += length

            if self.fscore > 0:
                gold_chunks = to_spans(gold, self.idx2label)
                gold_count += len(gold_chunks)

                guess_chunks = to_spans(sentence, self.idx2label)
                guess_count += len(guess_chunks)
            
                overlap_chunks = gold_chunks & guess_chunks
                overlap_count += len(overlap_chunks)

            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                idx = batch["id"][b]
                txt = txts[idx]
                self._write_sentence_conll(handle, sentence, gold, txt) 

        return correct_labels, total_labels, overlap_count, gold_count, guess_count

    def test(self, ts, batchsz=1, phase='Test', conll_file=None, txts=None):

        total_correct = total_sum = fscore = 0
        total_gold_count = total_guess_count = total_overlap_count = 0
        start_time = time.time()
    
        steps = int(math.floor(len(ts)/float(batchsz)))

        # Only if they provide a file and the raw txts, we can write CONLL file
        handle = None
        if conll_file is not None and txts is not None:
            handle = open(conll_file, "w")

        for i in range(steps):
            ts_i = batch(ts, i, batchsz)
            correct, count, overlaps, golds, guesses = self._batch(ts_i, handle, txts)
            total_correct += correct
            total_sum += count
            total_gold_count += golds
            total_guess_count += guesses
            total_overlap_count += overlaps

        duration = time.time() - start_time
        total_acc = total_correct / float(total_sum)

        # Only show the fscore if requested
        if self.fscore > 0:
            fscore = f_score(total_overlap_count,
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

        if handle is not None:
            handle.close()

        return total_acc, fscore

class Trainer:

    def __init__(self, sess, model, outdir, optim, eta, idx2label, fscore=0):
        
        self.sess = sess
        self.outdir = outdir
        self.loss = model.create_loss()
        self.model = model
        # Own this during training
        self.evaluator = Evaluator(sess, model, idx2label, fscore)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.MomentumOptimizer(eta, 0.9)

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

    def test(self, ts, batchsz, phase='Test', conll_file=None, txts=None):
        return self.evaluator.test(ts, batchsz, phase, conll_file, txts)
