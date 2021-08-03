import logging
import tensorflow as tf
from eight_mile.utils import get_version, exporter, register
from eight_mile.optz import create_lr_scheduler, register_lr_scheduler, MEAD_LAYERS_LR_SCHEDULERS
from eight_mile.optz import (
    LearningRateScheduler,
    ConstantScheduler,
    WarmupLearningRateScheduler,
    WarmupLinearScheduler,
    CyclicLRScheduler,
    PiecewiseDecayScheduler,
    ZarembaDecayScheduler,
    CosineDecayScheduler,
)

logger = logging.getLogger("mead.layers")


@register_lr_scheduler("default")
class ConstantSchedulerTensorFlow(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, global_step):
        tf.summary.scalar(name='lr', data=self.lr, step=tf.cast(global_step, tf.int64))
        return self.lr

    def __str__(self):
        return type(self).__name__ + "()"


@register_lr_scheduler("warmup_linear")
class WarmupLinearSchedulerTensorFlow(
    WarmupLearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, warmup_steps=16000, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        kwargs["lr"] = lr
        super().__init__(warmup_steps=warmup_steps, **kwargs)

    def __call__(self, global_step):
        new_lr = tf.minimum(1.0, global_step / float(self.warmup_steps)) * self.lr
        tf.summary.scalar(name='warmup_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


@register_lr_scheduler("clr")
class CyclicLRSchedulerTensorFlow(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr=1e-2, decay_steps=1000, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        kwargs["lr"] = lr
        super().__init__(**kwargs)
        self.max_lr = max_lr
        self.decay_steps = decay_steps

    def __call__(self, global_step):
        gs_f = tf.cast(global_step, tf.float32)
        cycle = tf.floor(1.0 + gs_f / (2.0 * self.decay_steps))
        x = tf.abs(gs_f / self.decay_steps - 2.0 * cycle + 1.0)
        clr = self.lr + (self.max_lr - self.lr) * tf.maximum(0.0, 1.0 - x)
        new_lr = tf.identity(clr, name="lr")
        tf.summary.scalar(name='clr_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


@register_lr_scheduler("sgdr")
class SGDRSchedulerTensorFlow(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, first_decay_steps=1000, **kwargs):
        super().__init__(**kwargs)
        self.first_decay_steps = first_decay_steps

    def __call__(self, global_step):
        new_lr = tf.identity(
            tf.compat.v1.train.cosine_decay_restarts(
                self.lr, global_step, first_decay_steps=self.first_decay_steps
            ),
            name="lr",
        )
        tf.summary.scalar(name='sgdr_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return type(self).__name__ + "()"


@register_lr_scheduler("zaremba")
class ZarembaDecaySchedulerTensorFlow(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    """Utility only, just to simplify the JSON"""

    def __init__(self, boundaries=None, decay_rate=None, **kwargs):
        lr = float(kwargs.get("lr", kwargs.get("eta", 1.0)))
        values = [lr / (float(decay_rate) ** i) for i in range(len(boundaries) + 1)]
        super().__init__(boundaries, values, kwargs.get("name"))


@register_lr_scheduler("composite")
class CompositeLRSchedulerTensorFlow(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warm=None, rest=None, plateau_steps=0, **kwargs):
        self.warm = warm
        self.rest = rest
        self.plateau_steps = plateau_steps

    def __call__(self, global_step):
        total_steps_lr1 = self.warm.warmup_steps + self.plateau_steps
        warm_tensor = self.warm(global_step)

        def call_warm():
            return warm_tensor

        rest_step = global_step - total_steps_lr1
        rest_tensor = self.rest(rest_step)

        def call_rest():
            return rest_tensor

        new_lr = tf.cond(global_step < total_steps_lr1, call_warm, call_rest)
        tf.summary.scalar(name='composite_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr

    def __str__(self):
        return "LRScheduler({}, {})".format(self.warm, self.rest)


@register_lr_scheduler("piecewise")
class PiecewiseDecaySchedulerTensorFlow(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    def __init__(self, boundaries, values, **kwargs):
        super().__init__(boundaries, values)

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='piecewise_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


@register_lr_scheduler("invtime")
class InverseTimeDecaySchedulerTensorFlow(tf.keras.optimizers.schedules.InverseTimeDecay):
    def __init__(self, decay_steps=16000, decay_rate=0.05, staircase=False, **kwargs):
        lr = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get("name"))

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='inv_time_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


@register_lr_scheduler("exponential")
class ExponentialDecaySchedulerTensorFlow(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, decay_steps=16000, decay_rate=0.5, staircase=False, **kwargs):
        lr = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(lr, decay_steps, decay_rate, staircase, kwargs.get("name"))

    def __call__(self, global_step):
        new_lr = super().__call__(global_step)
        tf.summary.scalar(name='exponential_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


@register_lr_scheduler("cosine")
class CosineDecaySchedulerTensorFlow(CosineDecayScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, decay_steps=16000, alpha=0.0, **kwargs):
        kwargs['lr'] = kwargs.get("lr", kwargs.get("eta", 0.01))
        super().__init__(decay_steps, alpha, **kwargs)

    def __call__(self, global_step):
        global_step = tf.math.minimum(global_step, self.decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(3.14159265 * global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        new_lr = self.lr * decayed
        tf.summary.scalar(name='cosine_decay_lr', data=new_lr, step=tf.cast(global_step, tf.int64))
        return new_lr


@register_lr_scheduler("linear")
class LinearDecaySchedulerTensorFlow(LearningRateScheduler, tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, decay_steps=1000, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, global_step):
        global_step = min(global_step, self.decay_steps)
        # Linear interpolation
        scaled_lr = self.lr * (1.0 - self.alpha) * (1.0 - global_step / self.decay_steps) + (self.alpha * self.lr)
        return scaled_lr


class EagerOptimizer:
    def __init__(self, loss, optimizer=None, **kwargs):
        self.loss = loss
        if "lr_function" in kwargs:
            lr_function = kwargs["lr_function"]
        else:
            if "lr_scheduler_type" not in kwargs:
                kwargs["lr_scheduler_type"] = "default"
            lr_function = create_lr_scheduler(**kwargs)
        # decay_fn = None
        # Right now this option is pointless since sparse updates dont work on GPU.  We just turn it off
        sgd_mom = float(kwargs.get("mom", 0.9))
        self.clip = kwargs.get("clip", 100)

        if optimizer:
            self.optimizer = optimizer
        else:
            optim = kwargs.get("optim", "sgd")
            lr = kwargs.get("lr", kwargs.get("eta", 0.01))

            if optim == "adadelta":
                rho = float(kwargs.get("rho", 0.95))
                eps = float(kwargs.get("epsilon", 1e-6))
                logger.info("adadelta(eta=%f, rho=%f, epsilon=%f)", lr, rho, eps)
                self.optimizer = tf.keras.optimizers.Adadelta(lr_function, rho, eps)
            elif optim == "adam":
                beta1 = float(kwargs.get("beta1", 0.9))
                beta2 = float(kwargs.get("beta2", 0.999))
                eps = float(kwargs.get("epsilon", 1e-8))
                logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
                self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)

            elif optim == "adamw":
                import tensorflow_addons as tfa
                beta1 = float(kwargs.get("beta1", 0.9))
                beta2 = float(kwargs.get("beta2", 0.999))
                eps = float(kwargs.get("epsilon", 1e-8))
                wd = float(kwargs.get("weight_decay", 0.0))
                if wd == 0.0:
                    logger.info("adam(eta=%f beta1=%f, beta2=%f, eps=%f)", lr, beta1, beta2, eps)
                    self.optimizer = tf.keras.optimizers.Adam(lr_function, beta1, beta2, eps)
                else:
                    def weight_decay_fn():
                        wd_t = lr_function(tf.cast(self.global_step, tf.float32) / lr) * wd
                        return wd_t
                    logger.info("adamw(eta=%f beta1=%f, beta2=%f, eps=%f, wd=%f)", lr, beta1, beta2, eps, wd)
                    self.optimizer = tfa.optimizers.AdamW(
                        weight_decay=weight_decay_fn, learning_rate=lr_function, beta_1=beta1, beta_2=beta2, epsilon=eps
                    )
            elif optim == "rmsprop":
                # Get mom again with difference default
                mom = float(kwargs.get("mom", 0.0))
                logger.info("rmsprop(eta=%f, mom=%f)", lr, mom)
                self.optimizer = tf.keras.optimizers.RMSprop(lr_function, momentum=mom)
            elif sgd_mom > 0:
                logger.info("sgd-mom(eta=%f, mom=%f)", lr, sgd_mom)
                self.optimizer = tf.keras.optimizers.SGD(lr_function, sgd_mom)
            else:
                logger.info("sgd(eta=%f)", lr)
                self.optimizer = tf.keras.optimizers.SGD(lr_function)

        logger.info("clip gradients at %s", self.clip)

    @property
    def global_step(self):
        return self.optimizer.iterations

    def update(self, model, x, y, num_replicas=1):
        with tf.GradientTape() as tape:
            loss_value = self.loss(model, x, y) / num_replicas
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value

    def update_with_hidden(self, model, h, x, y):
        with tf.GradientTape() as tape:
            loss_value, h = self.loss(model, h, x, y)

        grads = tape.gradient(loss_value, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value, h
