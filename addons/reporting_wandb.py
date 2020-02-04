from baseline.reporting import ReportingHook
from baseline.reporting import register_reporting
import wandb


@register_reporting(name="wandb")
class WandbReporting(ReportingHook):
    """Log results to wandb.
    
    pip install wandb
    create a profile with wandb: https://wandb.auth0.com/
    wandb login
    more info: https://github.com/wandb/examples
    """
    def __init__(self, **kwargs):
        super(WandbReporting, self).__init__(**kwargs)
        """"
        first do `wandb init` on terminal.
        """
        wandb.init()

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        metrics = {'{}/{}'.format(phase,key):metrics[key] for key in metrics}
        wandb.log(metrics)

