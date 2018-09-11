from baseline.tf.tfy import *
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer
import os


class LanguageModelTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerTf, self).__init__()
        self.model = model
        self.loss = model.create_loss()
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-lm-%d/lm" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-lm-%d" % os.getpid())
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def train(self, ts, reporting_fns):
        total_loss = 0.0
        iters = 0

        xfer_state = hasattr(self.model, 'initial_state')
        if xfer_state:
            state = self.model.sess.run(self.model.initial_state)

        fetches = {
            "loss": self.loss,
            "train_op": self.train_op,
            "global_step": self.global_step}

        if xfer_state:
            fetches["final_state"] = self.model.final_state
        step = 0
        metrics = {}

        for batch_dict in ts:

            feed_dict = self.model.make_input(batch_dict, True)
            if xfer_state:
                for i, (c, h) in enumerate(self.model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]

            if xfer_state:
                state = vals["final_state"]
            global_step = vals["global_step"]
            total_loss += loss
            iters += self.model.mxlen
            step += 1
            if step % 500 == 0:
                print(total_loss, iters)
                metrics['avg_loss'] = total_loss / iters
                metrics['perplexity'] = np.exp(total_loss / iters)
                for reporting in reporting_fns:
                    reporting(metrics, global_step, 'Train')

        metrics['avg_loss'] = total_loss / iters
        metrics['perplexity'] = np.exp(total_loss / iters)

        for reporting in reporting_fns:
            reporting(metrics, global_step, 'Train')
        return metrics

    def test(self, ts, reporting_fns, phase):
        total_loss = 0.0
        iters = 0
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs
        xfer_state = hasattr(self.model, 'initial_state')

        if xfer_state:
            state = self.model.sess.run(self.model.initial_state)

        fetches = {
            "loss": self.loss,
        }

        if xfer_state:
            fetches["final_state"] = self.model.final_state

        step = 0
        metrics = {}

        for batch_dict in ts:

            feed_dict = self.model.make_input(batch_dict, False)
            if xfer_state:
                for i, (c, h) in enumerate(self.model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]
            if xfer_state:
                state = vals["final_state"]
            total_loss += loss
            iters += self.model.mxlen
            step += 1

        metrics['avg_loss'] = total_loss / iters
        metrics['perplexity'] = np.exp(total_loss / iters)

        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics


def fit(model, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file(kwargs, 'lm', 'tf')
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(LanguageModelTrainerTf, model, **kwargs)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    model.save_using(saver)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif test_metrics[early_stopping_metric] < min_metric:
            last_improved = epoch
            min_metric = test_metrics[early_stopping_metric]
            print('New min %.3f' % min_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on min_metric %.3f at epoch %d' % (min_metric, last_improved))
    if es is not None:
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test')


