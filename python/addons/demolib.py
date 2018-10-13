import pandas as pd
from baseline.reader import register_reader, SeqLabelReader
from baseline.train import register_trainer, Trainer
from baseline.tf import *
from collections import Counter
from baseline.data import ExampleDataFeed, DictExamples
from baseline.confusion import ConfusionMatrix


@register_embeddings(name='cbow')
class CharBoWEmbeddings(TensorFlowEmbeddings):
    """Bag of character embeddings, sum char embeds, so in this case `wsz == dsz`

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharBoWEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharBoWLUT'.format(self.name))
        self.weights = kwargs.get('weights')
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        self.params = kwargs

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self, x=None):
        if x is None:
            x = CharBoWEmbeddings.create_placeholder(self.name)
        self.x = x
        return tf.reduce_sum(embed(x,
                                   self.get_vsz(),
                                   self.get_dsz(),
                                   tf.constant_initializer(self.weights, dtype=tf.float32),
                                   self.finetune,
                                   self.scope), axis=2, keep_dims=False)

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.dsz


@register_trainer(name='nsteps')
class NStepProgressClassifyTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        from baseline.tf.tfy import _add_ema
        super(NStepProgressClassifyTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.test_loss = model.create_test_loss()
        self.model = model
        self.nsteps = kwargs.get('nsteps', 10)
        self.global_step, train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            self.ema = True
            ema_op, self.ema_load, self.ema_restore = _add_ema(model, float(decay))
            with tf.control_dependencies([ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.ema = False
            self.train_op = train_op

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):

        if self.ema:
            self.sess.run(self.ema_load)

        cm = ConfusionMatrix(self.model.labels)
        steps = len(vs)
        total_loss = 0
        for batch_dict in vs:
            y = batch_dict['y']
            feed_dict = self.model.make_input(batch_dict)
            guess, lossv = self.sess.run([self.model.best, self.test_loss], feed_dict=feed_dict)
            total_loss += lossv
            cm.add_batch(y, guess)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)

        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics

    def train(self, loader, reporting_fns):

        if self.ema:
            self.sess.run(self.ema_restore)

        total_loss = 0
        steps = len(loader)
        for batch_dict in loader:
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            if step % self.nsteps == 0:
                metrics = {'avg_loss': total_loss/float(steps)}
                for reporting in reporting_fns:
                    reporting(metrics, step, 'Train')

        metrics = {}
        return metrics

    def checkpoint(self):
        self.model.saver.save(self.sess, "./tf-classify-%d/classify" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-classify-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

