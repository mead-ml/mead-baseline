import tensorflow as tf
import numpy as np
import time

class Trainer:

    def __init__(self, model, optim, eta):
        
        self.loss, self.errs, self.tot = model.createLoss()
        self.model = model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

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

    def train(self, ts, sess, summary_writer, dropout):
        total = 0
        total_loss = 0
        total_err = 0
        seq = np.random.permutation(len(ts))
        start_time = time.time()
    
        for j in seq:
            feed_dict = self.model.ex2dict(ts[j], 1.0-dropout)
        
            _, step, summary_str, lossv, errv, totv = sess.run([self.train_op, self.global_step, self.summary_op, self.loss, self.errs, self.tot], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            #print(lossv, errv, totv)
            total_loss += lossv
            total_err += errv
            total += totv

        duration = time.time() - start_time
        
        acc = 1.0 - (total_err/total)
            
        print('Train (Loss %.4f, Acc %.4f) (%.3f sec)' % (float(total_loss)/len(seq), acc, duration))


    def test(self, ts, sess):

        total_loss = total_err = total = 0
        start_time = time.time()
        for j in range(len(ts)):
            feed_dict = self.model.ex2dict(ts[j], 1.0)
            lossv, errv, totv = sess.run([self.loss, self.errs, self.tot], feed_dict=feed_dict)
            total_loss += lossv
            total_err += errv
            total += totv
        
        duration = time.time() - start_time

        err = total_err/total

        print('Test (Loss %.4f, Acc %.4f) (%.3f sec)' % (float(total_loss)/len(ts), 1.0 - err, duration))

        return err
