import eight_mile.progress
from eight_mile.progress import *
from eight_mile.utils import exporter, optional_params, register

__all__ = []
__all__.extend(eight_mile.progress.__all__)
export = exporter(__all__)

MEAD_LAYERS_PROGRESS = {}


@export
@optional_params
def register_progress(cls, name=None):
    return register(cls, MEAD_LAYERS_PROGRESS, name, 'progress')


@export
def create_progress_bar(steps, name='default', **kwargs):
    return MEAD_LAYERS_PROGRESS[name](steps, **kwargs)


@register_progress('jupyter')
class ProgressBarJupyterBaseline(ProgressBarJupyter): pass


@register_progress('default')
class ProgressBarTerminalBaseline(ProgressBarTerminal): pass


@register_progress('tqdm')
class ProgressBarTQDM(Progress):
    """Simple TQDM progress bar

    Writes a progress bar to an ipython widget

    """

    def __init__(self, total):
        super().__init__()
        from tqdm import tqdm

        self.progress = tqdm(total=total)

    def update(self, step=1):
        self.progress.update(step)

    def done(self):
        """Close the widget
        """
        self.progress.close()
