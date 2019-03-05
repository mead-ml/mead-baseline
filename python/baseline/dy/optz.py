import logging
import dynet as dy
from baseline.train import (
    register_lr_scheduler,
    create_lr_scheduler,
    ConstantScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
    InverseTimeDecayScheduler,
    ExponentialDecayScheduler,
    CompositeLRScheduler,
)

logger = logging.getLogger('baseline')


@register_lr_scheduler(name='default')
class ConstantSchedulerDyNet(ConstantScheduler):

    def __init__(self, **kwargs):
        super(ConstantSchedulerDyNet, self).__init__(**kwargs)


@register_lr_scheduler(name='warmup_linear')
class WarmupLinearSchedulerDyNet(WarmupLinearScheduler):

    def __init__(self, *args, **kwargs):
        super(WarmupLinearSchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='clr')
class CyclicLRSchedulerDyNet(CyclicLRScheduler):

    def __init__(self, *args, **kwargs):
        super(CyclicLRSchedulerDyNet, self).__init(*args, **kwargs)


@register_lr_scheduler(name='piecewise')
class PiecewiseDecaySchedulerDyNet(PiecewiseDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(PiecewiseDecaySchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='zaremba')
class ZarembaDecaySchedulerDyNet(ZarembaDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(ZarembaDecaySchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='cosine')
class CosineDecaySchedulerDyNet(CosineDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(CosineDecaySchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='invtime')
class InverseTimeDecaySchedulerDyNet(InverseTimeDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(InverseTimeDecaySchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='exponential')
class ExponentialDecaySchedulerDyNet(ExponentialDecayScheduler):
    def __init__(self, *args, **kwargs):
        super(ExponentialDecaySchedulerDyNet, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='composite')
class CompositeLRSchedulerDyNet(CompositeLRScheduler): pass


class OptimizerManager(object):

    def __init__(self, model, global_step=0, **kwargs):
        self.global_step = global_step
        if 'lr_scheduler_type' not in kwargs:
            kwargs['lr_scheduler_type'] = 'default'
        self.lr_function = create_lr_scheduler(**kwargs)
        self._init_optimizer(model, **kwargs)

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def _init_optimizer(self, model, **kwargs):
        mom = float(kwargs.get('mom', 0.0))
        optim = kwargs.get('optim', 'sgd')
        clip = kwargs.get('clip')

        self.current_lr = kwargs.get('eta', kwargs.get('lr', 0.01))
        if optim == 'adadelta':
            self.optimizer = dy.AdadeltaTrainer(model.pc)
        elif optim == 'adam':
            self.optimizer = dy.AdamTrainer(model.pc, alpha=self.current_lr, beta_1=kwargs.get('beta1', 0.9), beta_2=kwargs.get('beta2', 0.999), eps=kwargs.get('epsilon', 1e-8))
        elif optim == 'rmsprop':
            self.optimizer = dy.RMSPropTrainer(model.pc, learning_rate=self.current_lr)
        else:
            if mom == 0 or mom is None:
                self.optimizer = dy.SimpleSGDTrainer(model.pc, learning_rate=self.current_lr)
            else:
                logging.info('Using mom %f', mom)
                self.optimizer = dy.MomentumSGDTrainer(model.pc, learning_rate=self.current_lr, mom=mom)
        if clip is not None:
            self.optimizer.set_clip_threshold(clip)
        self.optimizer.set_sparse_updates(False)

    def _identity(self, _):
        return self.current_lr

    def update(self):
        """Runs at every step and updates the learning rate

        :return:
        """
        self.optimizer.update()
        self.current_lr = self.lr_function(self.global_step)
        self.optimizer.learning_rate = self.current_lr
        self.global_step += 1

    def zero_grad(self):
        self.optimizer.zero_grad()
