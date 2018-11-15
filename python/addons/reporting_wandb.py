from baseline.reporting import ReportingHook
from baseline.reporting import register_reporting
import wandb


@register_reporting(name="wandb")
class WandbReporting(ReportingHook):
    """Log results to tensorboard.

    Writes tensorboard logs to a directory specified in the `mead-settings`
    section for tensorboard. Otherwise it defaults to `runs`.
    """
    def __init__(self, **kwargs):
        super(WandbReporting, self).__init__(**kwargs)
        """"
        -  create a profile with wandb: https://wandb.auth0.com/
        -  wandb init
        """
        wandb.init()

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        wandb.log(metrics)

