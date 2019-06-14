import os
import time
import logging
import numpy as np
import tensorflow as tf
from baseline.tf.optz import optimizer
from baseline.progress import create_progress_bar
from baseline.utils import (
    listify,
    get_model_file,
    get_metric_cmp,
    convert_seq2seq_golds,
    convert_seq2seq_preds,
)
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.bleu import bleu
from baseline.tf.tfy import SET_TRAIN_FLAG, create_session


logger = logging.getLogger('baseline')


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerTf, self).__init__()
        self.sess = model.sess
        if kwargs.get('eval_mode', False):
            # When reloading a model things like `decoder.preds` used in the loss are not present.
            # Trying to create the loss would break things.
            self.loss = tf.no_op()
            self.test_loss = tf.no_op()
        else:
             self.loss = model.create_loss()
             self.test_loss = model.create_test_loss()

        self.model = model
        self.tgt_rlut = kwargs['tgt_rlut']
        self.base_dir = kwargs['basedir']
        if kwargs.get('eval_mode', False):
            # When reloaded a model creating the training op will break things.
            self.train_op = tf.no_op()
        else:
            self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self.beam = kwargs.get('beam', 10)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-seq2seq-%d/seq2seq" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = os.path.join(self.base_dir, 'seq2seq-model-tf-%d' % os.getpid())
        logger.info('Reloading %s', latest)
        g = tf.Graph()
        with g.as_default():
            SET_TRAIN_FLAG(None)
            sess = create_session()
            self.model = self.model.load(latest, predict=True, beam=self.beam, session=sess)

    def prepare(self, saver):
        self.model.set_saver(saver)

    def _num_toks(self, lens):
        return np.sum(lens)

    def calc_metrics(self, agg, norm):
        metrics = super(Seq2SeqTrainerTf, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def train(self, ts, reporting_fns):
        epoch_loss = 0
        epoch_toks = 0

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, True)
            _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            toks = self._num_toks(batch_dict['tgt_lengths'])
            report_loss = lossv * toks

            epoch_loss += report_loss
            epoch_toks += toks
            self.nstep_agg += report_loss
            self.nstep_div += toks

            if (global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def _evaluate(self, es, reporting_fns):
        """Run the model with beam search and report Bleu."""
        pg = create_progress_bar(len(es))
        preds = []
        golds = []
        start = time.time()
        for batch_dict in pg(es):
            tgt = batch_dict.pop('tgt')
            tgt_lens = batch_dict.pop('tgt_lengths')
            pred = [p[0] for p in self.model.predict(batch_dict)]
            preds.extend(convert_seq2seq_preds(pred, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        if phase == 'Test':
            return self._evaluate(vs, reporting_fns)
        self.valid_epochs += 1

        total_loss = 0
        total_toks = 0
        metrics = {}
        preds = []
        golds = []

        start = time.time()
        pg = create_progress_bar(len(vs))
        for batch_dict in pg(vs):

            feed_dict = self.model.make_input(batch_dict)
            lossv, top_preds = self.model.sess.run([self.test_loss, self.model.decoder.best], feed_dict=feed_dict)
            toks = self._num_toks(batch_dict['tgt_lengths'])
            total_loss += lossv * toks
            total_toks += toks

            preds.extend(convert_seq2seq_preds(top_preds.T, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(batch_dict['tgt'], batch_dict['tgt_lengths'], self.tgt_rlut))

        metrics = self.calc_metrics(total_loss, total_toks)
        metrics['bleu'] = bleu(preds, golds)[0]
        self.report(
            self.valid_epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq')
def fit(model, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model, **kwargs)
    m = model.replicas[0] if hasattr(model, 'replicas') else model
    feed_dict = {k: v for e in m.src_embeddings.values() for k, v in e.get_feed_dict().items()}
    feed_dict.update(m.tgt_embedding.get_feed_dict())
    init = tf.global_variables_initializer()
    model.sess.run(init, feed_dict)
    saver = tf.train.Saver()
    trainer.prepare(saver)

    checkpoint = kwargs.get('checkpoint')
    if checkpoint is not None:
        latest = tf.train.latest_checkpoint(checkpoint)
        print('Reloading ' + latest)
        model.saver.restore(model.sess, latest)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'bleu')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)
    if es is not None:
        trainer.recover_last_checkpoint()
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics
