"""MEAD3 core functions

This module contains development code which we hope will ultimately drive MEAD3 PyTorch
functionality.  Rather than dragging this functionality into core and phasing it in from
there, we can incubate it here as an addon.  For all of the registration alternatives,
the `*_type` here would be `mead3`

For training, MEAD3 will use the Trainer class which abstracts away device logic and supports
distributed processing. To facilitate this, instead of incorporating 2 hooks for training, overloading
the `Trainer` object and the `fit_func`, moving forward we will use a common `Trainer` object
for each backend and only support overloading fit functions.  This aims to simplify the codebase
while offering a lot more options for processing via the 8mile `Trainer`

In previous versions of MEAD, the trainers typically delegated some of the unpacking
duties to the models themselves.  This is undesirable, but to not do so added complications
to the trainers which otherwise were fairly general purpose.  We expected many use-cases
to necessitate model overloads, and at the same time we wanted to ensure maximum reusability
of the trainer (IOW to minimize the amount of trainer overloads), so the logic for unpacking
the prediction targets, for example, ended up in a function on the model (`make_input`).

By introducing a TrainingTarget class that is expected to be overloaded, we accomplish much
the same thing as before, but we encapsulate away the training logic from the model, at the
expense of requiring another possible overload to the user, but with a much smaller and more
compact scope.  When considering this tradeoff, we felt that it would not be too onerous to
the end user to construct one more custom object on occasions where it was warranted.

The net effect is that it should almost never be necessary now to overload the trainer itself


"""
import torch
from eight_mile.utils import revlut, to_spans, write_sentence_conll
from eight_mile.confusion import ConfusionMatrix
from eight_mile.pytorch.embeddings import *
from eight_mile.pytorch.optz import OptimizerManager
from eight_mile.train import LogAllMetrics, SpanF1Metric
from eight_mile.pytorch.train import TrainingTarget, Trainer
from baseline.model import create_model_for
from baseline.train import register_training_func, register_train_target, create_train_target


@register_train_target(task='classify', name='default')
class ClassifyTarget(TrainingTarget):
    """The "target" object of Trainer

    The Trainer's goal is ultimately to encapsulate the training of a type of model,
    but in doing so, it often needs specific logic related to the training problem.
    The `TrainingTarget` offers the extensibility needed to solve the full problem,
    while wrapping the model itself

    The `TrainingTarget` here also handles the loss function, which needs to be defined
    in order to train the model, along with the input step required to format the input
    data + the output labels
    """

    def __init__(self, model, **kwargs):
        super().__init__()
        # For MEAD models, the loss is often available as part of the model

        loss = model.create_loss(**kwargs)
        self._loss = loss
        self._model = model

    def train_step(self, batch):
        """Perform a step of training on a batch, computing loss and metrics

        :param batch: The raw batch from the reader
        :return: Step metrics
        """

        inputs = self._model.make_input(batch)
        truth = inputs.pop('y')

        pred = self._model(inputs)
        loss = self._loss(pred, truth)
        batchsz = truth.shape[0]
        report_loss = loss.item() * batchsz
        metrics = {'loss': loss, 'report_loss': report_loss}
        return metrics

    def eval_step(self, batch):
        """Perform a step of evaluation on a batch, computing loss and metrics

        :param batch: The raw batch from the reader
        :return: Step metrics
        """
        with torch.no_grad():
            cm = ConfusionMatrix(np.arange(self._model.num_classes))
            inputs = self._model.make_input(batch)
            truth = inputs.pop('y')

            pred = self._model(inputs)
            loss = self._loss(pred, truth)
            # If the truth is actually a prob dist, do an argmax RQ
            if truth.dtype == pred.dtype and len(truth.shape) == len(pred.shape):
                truth = torch.argmax(truth, -1)
            cm.add_batch(truth.cpu().numpy(), torch.argmax(pred, -1).cpu().numpy())
            batchsz = truth.shape[0]
            report_loss = loss.item() * batchsz
            metrics = {'loss': loss, 'report_loss': report_loss, 'confusion': cm}
        return metrics

    @property
    def model(self):
        return self._model


@register_training_func('classify', 'mead3')
def fit_classify_8mi(model, ts, vs, es, **kwargs):

    kwargs['lr'] = float(kwargs.get('lr', kwargs.get('eta', 0.001)))
    epochs = int(kwargs.get('epochs', 20))
    grad_accum = int(kwargs.get('grad_accum', 1))
    nstep = int(kwargs.get('nstep', 500))
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    distributed = bool(kwargs.get('distributed', False))
    local_rank = int(kwargs.get('local_rank', -1))
    #num_loader_workers = int(kwargs.get('num_loader_workers', 0))
    #pin_memory = bool(kwargs.get('pin_memory', True))

    #if not isinstance(ts, DataLoader):
    #    ts = DataLoader(ts, num_workers=num_loader_workers, batch_size=None, pin_memory=pin_memory)
    #if not isinstance(vs, DataLoader):
    #    vs = DataLoader(vs, batch_size=None, pin_memory=pin_memory)
    #if es and not isinstance(es, DataLoader):
    #    es = DataLoader(es, batch_size=None, pin_memory=pin_memory)

    early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
    if early_stopping_metric == 'none' or not early_stopping_metric:
        early_stopping_metric = None
    patience = kwargs.get('patience', epochs)
    if early_stopping_metric:
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    if type(model) is dict:
        checkpoint = kwargs.get('checkpoint')
        if checkpoint:
            model['checkpoint'] = checkpoint
        model = create_model_for('classify', **model)

    train_module = create_train_target(model, **kwargs)

    t = Trainer(train_module,
                train_metric_observers=LogAllMetrics("train"),
                valid_metric_observers=LogAllMetrics("valid"),
                test_metric_observers=LogAllMetrics("test"),
                **kwargs,
                )

    t.run(ts, vs, es, early_stopping_metric=early_stopping_metric,
          num_epochs=epochs, device=device,
          local_rank=local_rank, distributed=distributed,
          report_on=nstep,
          grad_accum=grad_accum)


@register_train_target(task='tagger', name='default')
class TaggerTarget(TrainingTarget):
    """The "target" object of Trainer

    The Trainer's goal is ultimately to encapsulate the training of a type of model,
    but in doing so, it often needs specific logic related to the training problem.
    The `TrainingTarget` offers the extensibility needed to solve the full problem,
    while wrapping the model itself

    The `TrainingTarget` here also handles the loss function, which needs to be defined
    in order to train the model, along with the input step required to format the input
    data + the output labels
    """

    def __init__(self, model, span_type=None, **kwargs):
        super().__init__()
        # For MEAD models, the loss is often available as part of the model
        self._model = model
        self.span_type = span_type
        self.idx2label = revlut(self.model.labels)

    def train_step(self, batch):
        """Perform a step of training on a batch, computing loss and metrics

        :param batch: The raw batch from the reader
        :return: Step metrics
        """

        inputs = self._model.make_input(batch)
        truth = inputs['y']
        loss = self._model.compute_loss(inputs)
        batchsz = truth.shape[0]
        report_loss = loss.item() * batchsz
        metrics = {'loss': loss, 'report_loss': report_loss}
        return metrics

    def process_output(self, guess, truth, sentence_lengths, ids, handle=None, txts=None):

        # For acc
        correct_labels = 0
        total_labels = 0
        truth_n = truth.cpu().numpy()
        # For f1
        gold_chunks = []
        pred_chunks = []

        # For each sentence
        for b in range(len(guess)):
            sentence = guess[b]
            if isinstance(sentence, torch.Tensor):
                sentence = sentence.cpu().numpy()
            sentence_length = sentence_lengths[b]
            gold = truth_n[b, :sentence_length]
            sentence = sentence[:sentence_length]

            valid_guess = sentence[gold != Offsets.PAD]
            valid_gold = gold[gold != Offsets.PAD]
            valid_sentence_length = np.sum(gold != Offsets.PAD)
            correct_labels += np.sum(np.equal(valid_guess, valid_gold))
            total_labels += valid_sentence_length
            gold_chunks.append(set(to_spans(valid_gold, self.idx2label, self.span_type)))
            pred_chunks.append(set(to_spans(valid_guess, self.idx2label, self.span_type)))

            # Should we write a file out?  If so, we have to have txts
            if handle is not None and txts is not None:
                txt_id = ids[b]
                txt = txts[txt_id]
                write_sentence_conll(handle, valid_guess, valid_gold, txt, self.idx2label)

        return correct_labels, total_labels, gold_chunks, pred_chunks

    def eval_step(self, batch):
        """Perform a step of evaluation on a batch, computing loss and metrics

        :param batch: The raw batch from the reader
        :return: Step metrics
        """
        with torch.no_grad():
            inputs = self.model.make_input(batch)
            truth = inputs.pop('y')
            lengths = inputs['lengths']
            ids = inputs['ids']
            with torch.no_grad():
                pred = self.model(inputs)
            correct, count, golds, guesses = self.process_output(pred, truth, lengths, ids)
            metrics = {'acc': (correct, count,), 'f1': SpanF1Metric((golds, guesses,))}
        return metrics

    @property
    def model(self):
        return self._model


@register_training_func('tagger', 'mead3')
def fit_tagger_8mi(model, ts, vs, es, **kwargs):

    kwargs['lr'] = float(kwargs.get('lr', kwargs.get('eta', 0.001)))
    epochs = int(kwargs.get('epochs', 20))
    grad_accum = int(kwargs.get('grad_accum', 1))
    nstep = int(kwargs.get('nstep', 500))
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    distributed = bool(kwargs.get('distributed', False))
    local_rank = int(kwargs.get('local_rank', -1))
    #num_loader_workers = int(kwargs.get('num_loader_workers', 0))
    #pin_memory = bool(kwargs.get('pin_memory', True))

    #if not isinstance(ts, DataLoader):
    #    ts = DataLoader(ts, num_workers=num_loader_workers, batch_size=None, pin_memory=pin_memory)
    #if not isinstance(vs, DataLoader):
    #    vs = DataLoader(vs, batch_size=None, pin_memory=pin_memory)
    #if es and not isinstance(es, DataLoader):
    #    es = DataLoader(es, batch_size=None, pin_memory=pin_memory)

    early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
    if early_stopping_metric == 'none' or not early_stopping_metric:
        early_stopping_metric = None
    patience = kwargs.get('patience', epochs)
    if early_stopping_metric:
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    if type(model) is dict:
        checkpoint = kwargs.get('checkpoint')
        if checkpoint:
            model['checkpoint'] = checkpoint
        model = create_model_for('tagger', **model)

    train_module = create_train_target(model, **kwargs)

    t = Trainer(train_module,
                train_metric_observers=LogAllMetrics("train"),
                valid_metric_observers=LogAllMetrics("valid"),
                test_metric_observers=LogAllMetrics("test"),
                **kwargs,
                )

    t.run(ts, vs, es, early_stopping_metric=early_stopping_metric,
          num_epochs=epochs, device=device,
          local_rank=local_rank, distributed=distributed,
          report_on=nstep,
          grad_accum=grad_accum)


