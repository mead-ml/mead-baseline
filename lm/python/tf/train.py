import numpy as np
import time
import tensorflow as tf
from data import batch


def zaremba_decay(eta, boundaries, decay_rate):

    values = [eta/(decay_rate**i) for i in range(len(boundaries))]
    print('Learning rate schedule:')
    print(boundaries)
    print(values)

    def _decay(lr, global_step):
        return tf.train.piecewise_constant(global_step, boundaries, values)
    return _decay


def exponential_staircase_decay(at_step=16000, decay_rate=0.5):

    def _decay(lr, global_step):
        return tf.train.exponential_decay(lr, global_step,
                                          at_step, decay_rate, staircase=True)
    return _decay


def optimizer(optim, eta, loss_fn, max_grad_norm, boundaries, decay_rate):
    global_step = tf.Variable(0, trainable=False)

    if optim == 'adadelta':
        optz = lambda lr: tf.train.AdadeltaOptimizer(lr, 0.95, 1e-6)

    elif optim == 'adam':
        optz = lambda lr: tf.train.AdamOptimizer(lr)
    else:
        optz = lambda lr: tf.train.GradientDescentOptimizer(lr)

    lr_decay_fn = zaremba_decay(eta, boundaries, decay_rate)
    return global_step, tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=max_grad_norm, learning_rate_decay_fn=lr_decay_fn)


class Trainer(object):

    def __init__(self, sess, model, outdir, optim, eta, max_grad_norm, boundaries, decay_rate):
        self.sess = sess
        self.model = model
        self.loss = model.create_loss()
        self.outdir = outdir
        self.global_step, self.train_op = optimizer(optim, eta, self.loss, float(max_grad_norm), boundaries, decay_rate)
        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.outdir + "/train", sess.graph)

    def checkpoint(self, name):
        self.model.saver.save(self.sess, self.outdir + "/train/" + name, global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint(self.outdir + "/train/")
        print("Reloading " + latest)
        self.model.saver.restore(self.sess, latest)

    def train(self, ts, pkeep):
        return self._run_epoch('Train', ts, pkeep, True)

    def test(self, ts, phase='Test'):
        return self._run_epoch(phase, ts, 1.0)

    def _run_epoch(self, phase, ts, pkeep, is_training=False):
        """Runs the model on the given data."""
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = self.sess.run(self.model.initial_state)

        fetches = {
            "loss": self.loss,
            "final_state": self.model.final_state,
        }
        if is_training:
            fetches["train_op"] = self.train_op
            fetches["global_step"] = self.global_step
            fetches["summary_str"] = self.summary_op

        step = 0

        nbptt = self.model.batch_info['nbptt']
        maxw = self.model.batch_info['maxw']
        batchsz = self.model.batch_info['batchsz']
        for next_batch in batch(ts, nbptt, batchsz, maxw):

            feed_dict = {
                self.model.x: next_batch[0],
                self.model.xch: next_batch[1],
                self.model.y: next_batch[2],
                self.model.pkeep: pkeep
            }
            for i, (c, h) in enumerate(self.model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = self.sess.run(fetches, feed_dict)
            cost = vals["loss"]
            state = vals["final_state"]
            if is_training:
                summary_str = vals["summary_str"]
                step = vals["global_step"]
                self.train_writer.add_summary(summary_str, step)
            costs += cost
            iters += nbptt
            step += 1
            if step % 500 == 0:
                print("step [%d] perplexity: %.3f" % (step, np.exp(costs / iters)))

        duration = time.time() - start_time
        avg_loss = costs / iters
        perplexity = np.exp(costs / iters)
        print('%s (Loss %.4f) (Perplexity = %.4f) (%.3f sec)' % (phase, avg_loss, perplexity, duration))
        return perplexity

