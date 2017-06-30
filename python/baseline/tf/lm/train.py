from baseline.tf.tfy import *
from baseline.utils import listify
from baseline.reporting import basic_reporting
import time

class LanguageModelTrainerTf:

    def __init__(self, model, **kwargs):
        self.model = model
        self.loss = model.create_loss()
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-checkpoints/lm", global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-checkpoints")
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def train(self, ts):
        return self._run_epoch(ts, True)

    def test(self, ts):
        return self._run_epoch(ts)

    def _run_epoch(self, ts, is_training=False):

        total_loss = 0.0
        iters = 0
        state = self.model.sess.run(self.model.initial_state)

        fetches = {
            "loss": self.loss,
            "final_state": self.model.final_state,
        }
        if is_training:
            fetches["train_op"] = self.train_op
            fetches["global_step"] = self.global_step

        step = 0
        metrics = {}

        for x, xch, y in ts:

            feed_dict = self.model.make_feed_dict(x, xch, y, not is_training)
            for i, (c, h) in enumerate(self.model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]
            state = vals["final_state"]
            total_loss += loss
            iters += self.model.nbptt
            step += 1
            if step % 500 == 0:
                # TODO make this a callback
                print("step [%d] perplexity: %.3f" % (step, np.exp(total_loss / iters)))

        metrics['avg_loss'] = total_loss / iters
        metrics['perplexity'] = np.exp(total_loss / iters)

        return metrics


def fit(model, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = kwargs['outfile'] if 'outfile' in kwargs and kwargs['outfile'] is not None else './seq2seq-model-tf'
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = LanguageModelTrainerTf(model, **kwargs)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    model.save_using(saver)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    for epoch in range(epochs):

        start_time = time.time()
        train_metrics = trainer.train(ts)
        train_duration = time.time() - start_time
        print('Training time (%.3f sec)' % train_duration)

        if after_train_fn is not None:
            after_train_fn(model)

        start_time = time.time()
        test_metrics = trainer.test(vs)
        test_duration = time.time() - start_time
        print('Validation time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(train_metrics, epoch, 'Train')
            reporting(test_metrics, epoch, 'Valid')

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
        start_time = time.time()
        trainer.test(es)
        test_duration = time.time() - start_time
        print('Test time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')