import time
import logging
from baseline.bleu import bleu
from baseline.progress import create_progress_bar
from baseline.utils import (
    listify,
    get_model_file,
    get_metric_cmp,
    convert_seq2seq_golds,
    convert_seq2seq_preds
)
from baseline.train import (
    Trainer,
    create_trainer,
    register_trainer,
    register_training_func
)
from baseline.dy.optz import *
from baseline.dy.dynety import *

logger = logging.getLogger('baseline')


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerDynet(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerDynet, self).__init__()
        self.model = model
        self.optimizer = OptimizerManager(model, **kwargs)
        self.tgt_rlut = kwargs['tgt_rlut']
        self.nsteps = kwargs.get('nsteps', 500)

    @staticmethod
    def _loss(outputs, labels, tgt_lengths):
        losses = [dy.pickneglogsoftmax_batch(out, label) for out, label in zip(outputs, labels)]
        mask, _ = sequence_mask(tgt_lengths, len(losses))
        mask = dy.transpose(mask)
        losses = dy.concatenate_cols(losses)
        masked_loss = dy.cmult(losses, mask)
        loss = dy.cdiv(dy.sum_batches(dy.sum_elems(masked_loss)), dy.sum_batches(dy.sum_elems(mask)))
        return loss

    @staticmethod
    def _num_toks(tgt_lens):
        return np.sum(tgt_lens)

    def calc_metrics(self, agg, norm):
        metrics = super(Seq2SeqTrainerDynet, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def train(self, loader, reporting_fns, **kwargs):
        self.model.train = True
        epoch_loss = 0.0
        epoch_toks = 0
        start = time.time()
        self.nstep_start = start
        for batch_dict in loader:
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            tgt = inputs.pop('tgt')
            tgt_lens = batch_dict['tgt_lengths']
            output = self.model.forward(inputs)
            loss = self._loss(output, tgt, tgt_lens)
            loss_val = loss.npvalue().item()
            loss.backward()
            self.optimizer.update()
            tok_count = self._num_toks(tgt_lens)
            report_loss = loss_val * tok_count
            epoch_loss += report_loss
            epoch_toks += tok_count
            self.nstep_agg += report_loss
            self.nstep_div += tok_count

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
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

    def test(self, vs, reporting_fns, phase):
        if phase == 'Test':
            return self._evaluate(vs, reporting_fns)
        self.model.train = False
        total_loss = total_toks = 0
        steps = len(vs)
        self.valid_epochs += 1

        preds = []
        golds = []

        start = time.time()
        pg = create_progress_bar(steps)
        for batch_dict in pg(vs):
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            tgt = inputs.pop('tgt')
            tgt_lens = batch_dict['tgt_lengths']
            output = self.model.forward(inputs)
            loss = self._loss(output, tgt, tgt_lens)
            toks = self._num_toks(tgt_lens)
            loss_val = loss.npvalue().item()
            total_loss += loss_val * toks
            total_toks += toks

            pred = [p[0] for p in self.model.predict(batch_dict, beam=1)]
            preds.extend(convert_seq2seq_preds(pred, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt.T, tgt_lens, self.tgt_rlut))

        metrics = self.calc_metrics(total_loss, total_toks)
        metrics['bleu'] = bleu(preds, golds)[0]
        self.report(
            self.valid_epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics

    def _evaluate(self, es, reporting_fns):
        self.model.train = False
        pg = create_progress_bar(len(es))
        preds = []
        golds = []
        start = time.time()
        for batch_dict in pg(es):
            tgt = batch_dict['tgt']
            tgt_lens = batch_dict['tgt_lengths']
            pred = [p[0] for p in self.model.predict(batch_dict)]
            preds.extend(convert_seq2seq_preds(pred, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq')
def fit(model, ts, vs, es=None, epochs=5, do_early_stopping=True,
        early_stopping_metric='bleu', **kwargs):

    patience = int(kwargs.get('patience', epochs))
    after_train_fn = kwargs.get('after_train_fn', None)

    model_file = get_model_file('seq2seq', 'dy', kwargs.get('basedir'))

    trainer = create_trainer(model, **kwargs)

    best_metric = 0
    if do_early_stopping:
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        logger.info("Doing early stopping on [%s] with patience [%d]", early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info("New best %.3f", best_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info("Stopping due to persistent failures to improve")
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = model.load(model_file)
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics
