import math
import logging
import torch
import torch.autograd
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
class ConstantSchedulerPyTorch(ConstantScheduler):

    def __init__(self, **kwargs):
        super(ConstantSchedulerPyTorch, self).__init__(**kwargs)


@register_lr_scheduler(name='warmup_linear')
class WarmupLinearSchedulerPyTorch(WarmupLinearScheduler):

    def __init__(self, *args, **kwargs):
        super(WarmupLinearSchedulerPyTorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='clr')
class CyclicLRSchedulerPyTorch(CyclicLRScheduler):

    def __init__(self, *args, **kwargs):
        super(CyclicLRSchedulerPyTorch, self).__init(*args, **kwargs)


@register_lr_scheduler(name='piecewise')
class PiecewiseDecaySchedulerPyTorch(PiecewiseDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(PiecewiseDecaySchedulerPyTorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='zaremba')
class ZarembaDecaySchedulerPyTorch(ZarembaDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(ZarembaDecaySchedulerPyTorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='cosine')
class CosineDecaySchedulerPyTorch(CosineDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(CosineDecaySchedulerPyTorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='invtime')
class InverseTimeDecaySchedulerPytorch(InverseTimeDecayScheduler):

    def __init__(self, *args, **kwargs):
        super(InverseTimeDecaySchedulerPytorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='exponential')
class ExponentialDecaySchedulerPyTorch(ExponentialDecayScheduler):
    def __init__(self, *args, **kwargs):
        super(ExponentialDecaySchedulerPyTorch, self).__init__(*args, **kwargs)


@register_lr_scheduler(name='composite')
class CompositeLRSchedulerPyTorch(CompositeLRScheduler):
    pass


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, set_lr, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
        self.set_lr = set_lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                lr = self.set_lr()
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

        return loss


class OptimizerManager(object):

    def __init__(self, model, global_step=0, **kwargs):
        self.global_step = global_step
        if 'lr_function' in kwargs:
            self.lr_function = kwargs['lr_function']
        else:
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
        wd = float(kwargs.get('weight_decay', 0))
        optim = kwargs.get('optim', 'sgd')
        self.current_lr = kwargs.get('eta', kwargs.get('lr', 0.01))
        self.step = self._step_then_update
        if optim == 'adadelta':
            logger.info('adadelta(eta=%f, wd=%d)', self.current_lr, wd)
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=self.current_lr, weight_decay=wd)
        elif optim.startswith('adam'):
            beta1 = kwargs.get('beta1', 0.9)
            beta2 = kwargs.get('beta2', 0.999)
            eps = kwargs.get('epsilon', 1e-8)
            if optim == 'adam':
                logger.info(
                    'adam(eta=%f, beta1=%f, beta2=%f, epsilon=%f, wd=%f)',
                    self.current_lr, beta1, beta2, eps, wd
                )
                self.optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.current_lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=wd
                )
            elif optim == 'adamw':
                logger.info(
                    'adamw(eta=%f, beta1=%f, beta2=%f, epsilon=%f, wd=%f)',
                    self.current_lr, beta1, beta2, eps, wd
                )
                self.optimizer = AdamW(
                    model.parameters(),
                    set_lr=self.update_lr,
                    lr=self.current_lr,
                    betas=(beta1, beta2),
                    eps=eps,
                    weight_decay=wd
                )
                self.step = self._step_and_update
        elif optim == 'rmsprop':
            mom = kwargs.get('mom', 0.0)
            logger.info('rmsprop(eta=%f, wd=%f, mom=%f)', self.current_lr, wd, mom)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.current_lr, weight_decay=wd, momentum=mom)
        elif optim == 'asgd':
            logger.info('asgd(eta=%f, wd=%f)', self.current_lr, wd)
            self.optimizer = torch.optim.ASGD(model.parameters(), lr=self.current_lr, weight_decay=wd)
        else:
            mom = kwargs.get('mom', 0.9)
            logger.info('sgd(eta=%f, mom=%f, wd=%f)', self.current_lr, mom, wd)
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.current_lr, momentum=mom, weight_decay=wd)

    def _identity(self, _):
        return self.current_lr

    def _step_and_update(self):
        """
        For AdamW, we need to do the LR update inside the optimizer before weight_decay, so have to make a custom path
        :return:
        """
        self.optimizer.step()
        self.global_step += 1

    def _step_then_update(self):
        """Runs at every step and updates the learning rate

        :return:
        """
        self.optimizer.step()
        self.current_lr = self.update_lr()
        self.global_step += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def update_lr(self):
        lr = self.lr_function(self.global_step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr

