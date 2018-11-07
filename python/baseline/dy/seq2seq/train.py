import time
import logging
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.dy.optz import *
from baseline.dy.dynety import *


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerDynet(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerDynet, self).__init__()
        self.model = model
        self.optimizer = OptimizerManager(model, **kwargs)
        self.log = logging.getLogger('baseline.timing')
        self.nsteps = kwargs.get('nsteps', 500)

    @staticmethod
    def _loss(outputs, labels, tgt_lengths):
        losses = [dy.pickneglogsoftmax_batch(out, label) for out, label in zip(outputs, labels)]
        mask, _ = sequence_mask(tgt_lengths, len(losses))
        mask = dy.transpose(mask)
        losses = dy.concatenate_cols(losses)
        masked_loss = dy.cmult(losses, mask)
        loss = dy.mean_batches(dy.cdiv(dy.sum_elems(masked_loss), dy.sum_elems(mask)))
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
            tok_count = self._num_toks(tgt)
            report_loss = loss_val * tok_count
            epoch_loss += report_loss
            epoch_toks += tok_count
            self.nstep_agg += report_loss
            self.nstep_div += tok_count

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns
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
        self.model.train = False
        total_loss = total_toks = 0
        steps = len(vs)
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

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

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq')
def fit(model, ts, vs, es=None, epochs=5, do_early_stopping=True,
        early_stopping_metric='avg_loss', **kwargs):

    patience = int(kwargs.get('patience', epochs))
    after_train_fn = kwargs.get('after_train_fn', None)

    model_file = get_model_file('seq2seq', 'dy', kwargs.get('basedir'))

    trainer = create_trainer(model, **kwargs)

    if do_early_stopping:
        print("Doing early stopping on [{}] with patience [{}]".format(early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    #if after_train_fn is not None:
    #    after_train_fn(model)
    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif test_metrics[early_stopping_metric] < min_metric:
            last_improved = epoch
            min_metric = test_metrics[early_stopping_metric]
            print("New min {:.3f}".format(min_metric))
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print("Stopping due to persistent failures to improve")
            break

    if do_early_stopping is True:
        print('Best performance on min_metric {:.3f} at epoch {}'.format(min_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        model = model.load(model_file)
        trainer.test(es, reporting_fns, phase='Test')
